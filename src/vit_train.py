
import sys
sys.path.append('.')

import torch
import torchvision
import importlib
import shutil
import uuid

import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import src.vit_data as vit_data_mod

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
import numpy as np
import torch


class ViTTrainer:

    def __init__(self, vit, config_name, dataloader, resume=False, resume_path=None):
        super().__init__()

        self.vit = vit
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.resume = resume
        self.resume_path = resume_path
        self.config = vit.config
        self.config_name = config_name

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
        self.lr = 0.001
        self.momentum = 0.9
        self.epochs = 2000000
        self.save_every = 100
        self.save_every_long = 1000

    def setup(self):

        self.timer = Timer()

        config = self.config
        vit = self.vit

        optimizer_name = config.OPTIMIZER

        if optimizer_name == 'SGD':
            self.optimizer = optim.SGD(vit.parameters(), lr=self.lr, momentum=self.momentum)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(vit.parameters(), lr=0.001)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")

        # Track progress
        if not self.resume:
            self.epochs_arr = []
            self.validation_losses = []
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

            self.out_dir = f"{config.OUT_DIR}_{today_str}_{random_hash}"
            mkdir_safe(self.out_dir)
            shutil.copyfile(f"config/{self.config_name}.py", f"{self.out_dir}/config.py")
            last_epoch = 0
        else:
            self.out_dir = self.resume_path
            vit.load_state_dict(torch.load(f'{self.resume_path}/model.torch',
                map_location=torch.device('cpu')))
            print_fl(f"Resuming from {last_epoch}...")

        self.epochs_to_run = np.arange(last_epoch+1, last_epoch+self.epochs)
        self.model_save_path = f"{self.out_dir}/model.torch"
        self.loss_path = f"{self.out_dir}/loss.csv"
        self.loss_fig_path = f"{self.out_dir}/loss.png"
        self.best_model_save_path = f"{self.out_dir}/model.best.torch"
        self.min_validation_loss = float('inf')
        self.min_validation_model = None


    def train(self):

        vit = self.vit
        epochs_arr = self.epochs_arr
        validation_losses = self.validation_losses
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

        if self.resume:
            losses_df = pd.read_csv(f"{self.resume_path}/loss.csv")
            debug_train = list(losses_df['debug_train'].values)
            debug_valid = list(losses_df['debug_valid'].values)
            debug_test = list(losses_df['debug_test'].values)
        else:
            debug_train = []
            debug_valid = []
            debug_test = []

        for epoch in self.epochs_to_run:

            running_loss = 0.0

            # Enumerate trainloader batches
            is_perturb = False
            for i, data in enumerate(trainloader, 0):

                images, tx, _, _, _ = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, weights = vit(images.float().to(device))
                loss = self.criterion(outputs.double(), tx.reshape(outputs.shape).to(device))
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

            if epoch % 10 == 0:

                validation_loss = self.compute_validation_loss(validationloader)

                # Save min loss model
                if not np.isnan(validation_loss) and validation_loss < self.min_validation_loss:
                    pd.DataFrame({'epoch': [epoch]}).to_csv(f"{self.out_dir}/model.best.epoch.txt", index=False)
                    torch.save(vit.state_dict(), best_model_save_path)
                    self.min_validation_loss = validation_loss
                    self.min_validation_model = vit.state_dict().copy()

                torch.save(vit.state_dict(), model_save_path)

                # Save intermediate model, more frequently early on to capture
                # validation loss minimum
                if ((epoch % self.save_every == 0 and epoch < self.save_every_long) or 
                    (epoch % self.save_every_long == 0 and epoch >= self.save_every_long)):
                   torch.save(vit.state_dict(), f"{model_save_path}.{epoch}")

                epochs_arr.append(epoch)
                validation_losses.append(validation_loss)
                train_losses.append(train_loss)

                self.compute_predictions_losses()

                perturb_str = "* Saddle conditions met! Perturbing model weights *" if is_perturb else ""
                print_fl('[%d] train loss: %.5f, test loss %.5f, %s %s' %
                      (epoch + 1, self.train_loss, self.test_loss, timer.get_time(), perturb_str))

                debug_train.append(self.train_loss)
                debug_valid.append(self.validation_loss)
                debug_test.append(self.test_loss)

                print_fl(self.loss_str)

                fig = self.plot_predictions()
                plt.savefig(f"{self.out_dir}/predictions.png", dpi=150)
                plt.close(fig)
                plt.cla()
                plt.clf()

                loss_df = pd.DataFrame({
                        'epoch': epochs_arr,
                        'train_loss': train_losses,
                        'validation_loss': validation_losses,
                        'debug_train': debug_train,
                        'debug_valid': debug_valid,
                        'debug_test': debug_test,
                    })

                # Plot loss
                fig = plot_loss_progress(loss_df, m=50)
                plt.savefig(self.loss_fig_path, dpi=150)
                plt.close(fig)
                plt.cla()
                plt.clf()

                self.loss_df = loss_df
                loss_df.to_csv(self.loss_path, index=False)
                
            # End early if loss is reached
            if train_loss < self.stoploss_value:
                print_fl(f"[{epoch}] Stop loss reached: {train_loss} < {self.stoploss_value}. Ending early.")
                break

        print_fl(f'Finished Training {timer.get_time()}')

    def plot_prediction_performance(self, all_tx, all_predictions, title, ax=None):
        
        unscaled_logTPM = np.log2(self.dataloader.dataset.unscaled_TPM+1)
        y = self.dataloader.dataset.unscale_log_tx(all_tx)
        x = self.dataloader.dataset.unscale_log_tx(all_predictions)

        r2 = r2_score(y, x)

        if ax is None:
            ax = plt.gca()

        plot_density_scatter(x, y, cmap='Spectral_r', bw=(0.25, 0.25), zorder=2, ax=ax)

        plt.plot([-20, 20], [-20, 20], c='black', linestyle='solid', lw=0.5, zorder=4)

        plt.xticks(np.arange(0, 20, 5))
        plt.yticks(np.arange(0, 20, 5))
        plt.xlim(-0.5, 15.5)
        plt.ylim(-0.5, 15.5)

        plt.ylabel('True log$_2$ transcript level, TPM')
        plt.xlabel('Predicted log$_2$ transcript level, TPM')

        plt.title(f"{title}, n={len(all_tx)}, $R^2$={r2:.3f}")


    def compute_predictions_losses(self, max_num=float('inf')):

        timer = Timer()
        print_fl("Computing test predictions", end='...')
        (self.test_tx, self.test_predictions, self.test_r2,
         self.test_loss) = self.generate_predicted_vs_true_data(self.dataloader.testloader, max_num)
        print_fl(f"Done. {self.test_loss:.3f}, {self.test_r2:.3f}")
        timer.print_time()

        print_fl("Computing train predictions", end='...')
        (self.train_tx, self.train_predictions, self.train_r2,
         self.train_loss) = self.generate_predicted_vs_true_data(self.dataloader.trainloader, max_num)
        print_fl(f"Done. {self.train_loss:.3f}, {self.train_r2:.3f}")
        timer.print_time()

        print_fl("Computing validation predictions", end='...')
        (self.validation_tx, self.validation_predictions, self.validation_r2, 
         self.validation_loss) = self.generate_predicted_vs_true_data(self.dataloader.validationloader, max_num)
        print_fl(f"Done. {self.validation_loss:.3f}, {self.validation_r2:.3f}")
        timer.print_time()

        perf_str = (f"Loss:\n"
                    f"  Train:\t{self.train_loss:.3f}\n  Valid:\t{self.validation_loss:.3f}"
                    f"\n  Test: \t{self.test_loss:.3f}"
                    f"\n\nR2:\n"
                    f"  Train:\t{self.train_r2:.3f}\n  Valid:\t{self.validation_r2:.3f}"
                    f"\n  Test: \t{self.test_r2:.3f}")

        self.perf_str = perf_str
        self.loss_str =  (f" Train: {self.train_loss:.3f}, Valid:\t{self.validation_loss:.3f},"
                          f" Test: \t{self.test_loss:.3f}")


    def plot_predictions(self):
        fig = plt.figure(figsize=(8, 3))

        plt.subplot(1, 2, 1)
        self.plot_prediction_performance(self.train_tx, self.train_predictions, 'Train')

        plt.subplot(1, 2, 2)
        self.plot_prediction_performance(self.test_tx, self.test_predictions, 'Test')

        return fig


    def generate_predicted_vs_true_data(self, dataloader, max_num=float('inf')):

        if len(dataloader.dataset) == 0:
            return None, None, np.nan, np.nan

        vit = self.vit
        device = self.device
        all_tx = np.array([])
        all_predictions = np.array([])
        i = 0
        running_loss = 0

        with torch.no_grad():
            for imgs, tx, _, _, _ in dataloader:    

                out, weights = vit(imgs.float().to(self.device))
                predictions = out.detach().to(torch.device('cpu')).numpy().flatten()    
                tx = tx.to(torch.device('cpu'))

                all_tx = np.concatenate([all_tx, tx])
                all_predictions = np.concatenate([all_predictions, predictions])
                
                loss = self.criterion(out.to(torch.device('cpu')), 
                                      tx.reshape(out.shape).to(torch.device('cpu')))
                running_loss += loss.item()
                i += 1

                # Subsample to save time
                if max_num is not None and len(all_predictions) > max_num:
                    break

        y = self.dataloader.dataset.unscale_log_tx(all_tx)
        x = self.dataloader.dataset.unscale_log_tx(all_predictions)

        r2 = r2_score(y, x)

        return all_tx, all_predictions, r2, (running_loss / i)


    def compute_validation_loss(self, dataloader, num_batches=-1):

        if len(dataloader.dataset) == 0:
            return np.nan

        device = self.device
        vit = self.vit
        
        with torch.no_grad():

            i = 0
            running_loss = 0
            for data in dataloader:
                images, tx, _, _, _ = data
                images = images.to(device)
                tx = tx.to(device)

                # calculate outputs by running images through the network
                outputs, weights = vit(images.float().to(device))

                outputs = outputs.to(device)
                loss = self.criterion(outputs, tx.reshape(outputs.shape))
                running_loss += loss.item()

                i+=1
                if i == num_batches: break

        return running_loss / i


    def compute_attentions(self, t=120):

            timer = Timer()
            vit_data = self.dataloader.dataset
            vit = self.vit
            n = len(vit_data)

            # TODO: divide by number of samples
            if t is None:
                m = n
            else:
                m = n//6

            collected_attentions = np.zeros((m, vit.get_patch_rows(), vit.get_patch_columns()))

            i = 0
            for cur_data in vit_data:

                # Skip 
                if (t is not None) and (cur_data[-1] != t): continue

                x = cur_data[0]

                att_mask = rollout(vit, x, discard_ratio=0.95, head_fusion='mean', 
                    device=torch.device('cpu'), attention_channel_idx=0)
                collected_attentions[i] = att_mask
                
                timer.print_label(f"{i+1}/{m}", conditional=(i % 1000 == 0))

                i += 1

            self.collected_attentions = collected_attentions

            return collected_attentions


    def plot_gene(self, gene_name, time):
        plot_gene_prediction(gene_name, time, self.vit, self.dataloader.dataset,
            orf_plotter=self.orf_plotter, rna_plotter=self.rna_plotter)


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    return device


def load_model_config(config, legacy=False):

    from src.vit_legacy import ViT as ViT_legacy
    from src.vit import ViT

    if legacy:
        model_class = ViT_legacy
    else:
        model_class = ViT

    vit = model_class(config)
    vit.legacy = legacy

    return vit


def load_model_dir(model_dir, legacy=False):
    config_path = f"{model_dir.replace('/', '.')}.config"
    config = importlib.import_module(config_path)
    vit = load_model_config(config, legacy=legacy)
    vit.load_state_dict(torch.load(f"{model_dir}/model.torch", map_location=torch.device('cpu')))
    return vit, config


def plot_loss_progress(loss_df, m=50):

    def _get_ylim(data):
        data_min = data.min()
        data_max = data.max()
        data_span = data_max-data_min
        return data_min-data_span*0.5, data_max+data_span*0.5

    fig = plt.figure(figsize=(14, 3))
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    plt.subplot(1, 3, 1)
    plt.plot(loss_df.epoch, loss_df.debug_train, label='Training loss')
    plt.plot(loss_df.epoch, loss_df.debug_test, label='Test loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(loss_df.epoch[-m:], loss_df.debug_train[-m:], label='Training loss')
    plt.title(f"Training loss, {loss_df.debug_train.values[-1]:.4f}")
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_df.epoch[-m:], loss_df.debug_test[-m:], label='Test loss',
        c=plt.get_cmap('tab10')(1))
    
    plt.title(f"Test loss, {loss_df.debug_test.values[-1]:.4f}, (min={loss_df.debug_test.values.min():.4f})")
    return fig


def main():

    # Load config from command-line
    if len(sys.argv) == 3:
        assert sys.argv[1] == 'resume'
        resume_path = sys.argv[2]
        resume = True
        vit, config = load_model_dir(resume_path)
        config_name = None

    elif len(sys.argv) == 2:
        config_name = sys.argv[1]
        config = importlib.import_module(f"config.{config_name}")
        vit = load_model_config(config)
        resume = False
        resume_path = None

    print_fl(f"Config: {vit.config_repr()}")

    # Data loading
    print_fl("Loading data...")

    # Dynamic load the correct data loading function
    dataset = getattr(vit_data_mod, config.DATA_FUNC)(replicate_mode=config.REPLICATE_MODE,
        predict_tpm=config.PREDICT_TPM)

    if not resume:
        dataloader = ViTDataLoader(dataset, batch_size=config.BATCH_SIZE, 
            split_type=config.SPLIT_TYPE, split_arg=config.SPLIT_ARG,
            valid_type=config.VALIDATION_TYPE, valid_arg=config.VALIDATION_ARG)
    else:
        data_indices_path = f"{resume_path}/indices.csv"
        dataloader = ViTDataLoader(dataset, batch_size=config.BATCH_SIZE, 
            split_type=config.SPLIT_TYPE, split_arg=config.SPLIT_ARG,
            valid_type=config.VALIDATION_TYPE, valid_arg=config.VALIDATION_ARG,
            indices_path=data_indices_path)

    print_fl(f"Dataloader split: {dataloader.split_repr()}")

    # Initialize trainer
    print_fl("Initializing trainer...")
    trainer = ViTTrainer(vit, config_name, dataloader, resume=resume, resume_path=resume_path)
    trainer.setup()
    print_fl(f"Writing to {trainer.out_dir}")

    if not resume:
        data_indices_path = f"{trainer.out_dir}/indices.csv"
        dataloader.save_indices(data_indices_path)

    # Train
    print_fl("Training...")
    trainer.train() 


if __name__ == '__main__':
    main()
