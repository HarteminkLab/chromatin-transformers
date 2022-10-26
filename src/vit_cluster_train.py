
import sys
sys.path.append('.')

import torch
import torchvision
import importlib
import shutil
import uuid
import torch
import pacmap

import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import src.vit_data as vit_data_mod
import numpy as np

from sklearn.cluster import KMeans
from datetime import date
from torch import nn
from torch import Tensor
from PIL import Image
from scipy.stats import spearmanr
from src.data_loader import ViTDataLoader
from src.timer import Timer
from src.utils import print_fl, mkdir_safe
from sklearn.metrics import r2_score
from src.plot_utils import plot_density_scatter
from src.vit_viz import rollout, plot_gene_prediction
from src.timer import Timer
from src.vit_data import load_cell_cycle_data
from src.vit_data_clustering import ViTDataDeepClustering
from src.vit_train import get_device, load_model_config


class ViTDeepClusterTrainer:

    def __init__(self, vit, config_name, dataloader, resume=False, resume_path=None, debug=False,
            criterion=nn.MultiLabelSoftMarginLoss):
        super().__init__()

        self.vit = vit
        self.dataloader = dataloader
        self.criterion = criterion()
        self.resume = resume
        self.resume_path = resume_path
        self.config = vit.config
        self.config_name = config_name
        self.debug = debug

        num_batches = len(dataloader.trainloader)

        self.device = get_device()
        self.vit = self.vit.to(self.device)

        # Training parameters
        # 
        # How many loss values to check for in train_losses array to detect saddle points, losses are
        # collected every 10 epochs. And threshold considered as "stuck" in saddle point.
        # perturbation settings if enabled
        self.last_k_perturb = 5
        self.loss_threshold = 1e-3

        # Value of training loss to stop training
        self.stoploss_value = 1e-5
        self.lr = vit.config.LR
        self.momentum = 0.9
        self.epochs = 2000000
        self.save_every = 100
        self.save_every_long = 1000
        self.print_every = 100

    def setup(self):

        self.timer = Timer()

        config = self.config
        vit = self.vit

        optimizer_name = config.OPTIMIZER

        if optimizer_name == 'SGD':
            self.optimizer = optim.SGD(vit.parameters(), lr=self.lr, momentum=self.momentum)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(vit.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")

        # Track progress
        if not self.resume:
            self.epochs_arr = []
            self.train_losses = []
            last_epoch = 0
        else:
            losses_df = pd.read_csv(f"{self.resume_path}/loss.csv")
            self.epochs_arr = list(losses_df['epoch'].values)
            self.train_losses = list(losses_df['train_loss'].values)
            self.validation_losses = list(losses_df['validation_loss'].values)
            last_epoch = self.epochs_arr[-1]

        if not self.resume:
            today = date.today()
            today_str = today.strftime("%Y%m%d")
            random_hash = uuid.uuid4().hex[0:4]

            if not self.debug:
                self.out_dir = f"{config.OUT_DIR}_{today_str}_{random_hash}"
            else:
                self.out_dir = f"{config.OUT_DIR}"

            mkdir_safe(self.out_dir)
            shutil.copyfile(f"config/{self.config_name}.py", f"{self.out_dir}/config.py")
            last_epoch = 0
        else:
            self.out_dir = self.resume_path
            vit.load_state_dict(torch.load(f'{self.resume_path}/model.torch',
                map_location=torch.device('cpu')))
            print_fl(f"Resuming from {last_epoch}...")

        
        self.model_save_path = f"{self.out_dir}/model.torch"
        self.loss_path = f"{self.out_dir}/loss.csv"
        self.loss_fig_path = f"{self.out_dir}/loss.png"
        self.best_model_save_path = f"{self.out_dir}/model.best.torch"
        self.min_validation_loss = float('inf')
        self.min_validation_model = None


    def train(self, total_epochs=100000, plot_results=False):

        self.epochs_arr = []
        self.train_losses = []
        self.proportions = {}
        for c in range(self.config.NUM_CLASSES):
            self.proportions[c] = []

        self.epochs_to_run = np.arange(1, total_epochs)
        vit = self.vit
        epochs_arr = self.epochs_arr
        train_losses = self.train_losses
        optimizer = self.optimizer
        device = self.device
        trainloader = self.dataloader.trainloader
        validationloader = self.dataloader.validationloader
        testloader = self.dataloader.testloader
        timer = self.timer
        config = self.config
        model_save_path = self.model_save_path
        last_k_perturb = self.last_k_perturb
        best_model_save_path = self.best_model_save_path

        # Compute image vector once for clustering
        from einops.layers.torch import Rearrange
        vectorize_grid = Rearrange('b i (r) (c) -> b (i r c)')
        self.all_img_vector = vectorize_grid(torch.Tensor(self.dataloader.trainloader.dataset[:][1]))

        # pacmap fig
        fig = None

        for epoch in self.epochs_to_run:

            running_loss = 0.0

            # Enumerate trainloader batches
            is_perturb = False
            for i, data in enumerate(trainloader, 0):

                indices, images, pseudo_labels = data
                one_hot_labels = torch.Tensor(one_hot_encode(pseudo_labels.numpy(), config.NUM_CLASSES))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                (label_weights, penult_weights), weights = vit(images.float().to(device))

                loss = self.criterion(label_weights.double(), 
                    one_hot_labels.reshape(label_weights.shape).to(device).double())
                loss.backward()

                # Apply perturbation to weights to knock out of saddle points
                if epoch % 10 == 0:
                    
                    if config.PERTURBATION > 0. and len(train_losses) > 1:

                        last_k_diff = abs(np.max(train_losses[-last_k_perturb:]) -
                                          np.min(train_losses[-last_k_perturb:]))

                        # If the last 30 epochs do not differ by the threshold and
                        # we're above the loss limit, we may be in a saddle point.
                        # Perturb the weights of the model
                        if ((last_k_diff < self.loss_threshold) and
                            (loss.item() > config.PERTURBATION_LOSS_LIM)):

                            is_perturb = True
                            with torch.no_grad():
                                for param in vit.parameters():
                                    perturb = np.random.normal(0, config.PERTURBATION, param.shape)
                                    param += torch.tensor(perturb).to(device)

                optimizer.step()

                running_loss += loss.item()

            train_loss = (running_loss / len(trainloader))

            if epoch % 2 == 0:
                label_counts, counts_str = self.dataloader.dataset.get_label_counts()
                self.train_losses.append(train_loss)
                self.epochs_arr.append(epoch)
                for c in label_counts.keys():
                    self.proportions[c].append(label_counts[c])

            param_sum = 0
            for p in vit.parameters():
                param_sum += p.sum().detach().numpy()

            # Clustering for pseudo labels
            self.cluster_and_assign_pseudo_labels()

            if epoch % self.print_every == 0:
                label_counts, counts_str = self.dataloader.dataset.get_label_counts()
                if plot_results and fig is not None:
                    plt.close(fig)
                    plt.cla()
                    plt.clf()

                print_fl('[%d] %.3f loss, %s' % (epoch+1, train_loss, counts_str))

                if plot_results:
                    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
                    self.plot_pacmap_labels(ax0)
                    ax1.plot(self.epochs_arr, self.train_losses)
                    plt.savefig(f"{self.out_dir}/progress.png")

                torch.save(vit.state_dict(), model_save_path)

            # Save intermediate model, more frequently early on to capture
            # validation loss minimum
            if ((epoch % self.save_every == 0 and epoch < self.save_every_long) or 
                (epoch % self.save_every_long == 0 and epoch >= self.save_every_long)):
               torch.save(vit.state_dict(), f"{model_save_path}.{epoch}")

        print_fl(f'Finished Training {timer.get_time()}')


    def plot_pacmap_labels(self, ax):
        vit = self.vit
        data = self.dataloader.dataset
        with torch.no_grad():
            indices, images, pseudo_labels = data[:]
            (label_weights, penult_weights), weights = vit(torch.Tensor(images))
            embedding = pacmap.PaCMAP(n_components=2)
            pacmap_embeddings = embedding.fit_transform(penult_weights.detach().numpy())

        for l in set(pseudo_labels):
            idx = (pseudo_labels == l)
            ax.scatter(pacmap_embeddings[idx, 0], pacmap_embeddings[idx, 1], 
                label=f"Cluster {l:.0f}", s=1)

        ax.legend()


    def cluster_and_assign_pseudo_labels(self):

        seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        k = self.config.NUM_CLASSES
        
        # Compute the label weights and penultimate layer's weights
        # as features for clustering
        with torch.no_grad():
            res = self.vit(torch.Tensor(self.dataloader.dataset[:][1]))
            (label_weights, pen_x), att_weights = res

        np.random.seed(123)
        kmeans = KMeans(n_clusters=k, random_state=123).fit(pen_x.detach().numpy())
        indices = np.arange(len(pen_x))

        # reassign the pseudo labels
        self.dataloader.dataset.pseudo_labels[indices] = kmeans.labels_


def one_hot_encode(labels, n_classes):

    classes_universe = np.arange(n_classes)

    mapping = {}
    for x in range(n_classes):
        mapping[classes_universe[x]] = x

    one_hot_encode = []

    for label in labels:
        arr = list(np.zeros(n_classes, dtype = int))
        arr[mapping[label]] = 1
        one_hot_encode.append(arr)

    return np.array(one_hot_encode)


def main():

    if len(sys.argv) == 2:
        config_name = sys.argv[1]
        config = importlib.import_module(f"config.{config_name}")
        vit = load_model_config(config)
        resume = False
        resume_path = None

    print_fl(f"Config: {vit.config_repr()}")

    # Data loading
    print_fl("Loading data...")
    vit_data = load_cell_cycle_data(config.REPLICATE_MODE, config.CHANNEL_1, config.PREDICT_TPM, 
                                    init_class=ViTDataDeepClustering, debug_n=4000)
    dataloader = ViTDataLoader(vit_data, batch_size=config.BATCH_SIZE, 
                split_type=config.SPLIT_TYPE, split_arg=config.SPLIT_ARG,
                valid_type=config.VALIDATION_TYPE, valid_arg=config.VALIDATION_ARG)

    trainer = ViTDeepClusterTrainer(vit, config_name, dataloader, resume=False, resume_path=None,
        criterion=nn.MultiLabelSoftMarginLoss)
    trainer.setup()
    print_fl(f"Writing to {trainer.out_dir}")

    # Train
    print_fl("Training...")
    trainer.train()


if __name__ == '__main__':
    main()
