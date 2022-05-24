
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

from datetime import date
from torch import nn
from torch import Tensor
from PIL import Image
from scipy.stats import spearmanr
from src.vit_data import load_cd_data
from src.data_loader import ViTDataLoader
from src.timer import Timer
from src.utils import print_fl, mkdir_safe
from src.vit import ViT


class ViTTrainer:

    def __init__(self, vit, dataloader):
        super().__init__()

        self.vit = vit
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.resume_path = None
        self.config = vit.config

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


        last_epoch = 0
        self.epochs = 100000
        self.epochs_to_run = np.arange(last_epoch+1, last_epoch+self.epochs)

    def setup(self):
    
        # print_fl(f"Starting training, {epochs} epochs with {num_batches} "
        #              f"batches of size {config.BATCH_SIZE}...")

        self.timer = Timer()

        config = self.config
        vit = self.vit

        optimizer_name = config.OPTIMIZER
        self.optimizer = optim.SGD(vit.parameters(), lr=self.lr, momentum=self.momentum)

        # Track progress
        self.epochs_arr = []
        self.validation_losses = []
        self.train_losses = []

        today = date.today()
        today_str = today.strftime("%Y%m%d")
        random_hash = uuid.uuid4().hex[0:4]

        self.out_dir = f"{config.OUT_DIR}_{today_str}_{random_hash}"

        loss_path = f"{self.out_dir}/loss.csv"
        loss_fig_path = f"{self.out_dir}/loss.png"


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

        for epoch in self.epochs_to_run:

            running_loss = 0.0

            # Enumerate trainloader batches
            for i, data in enumerate(trainloader, 0):

                images, tx, _, _, _ = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, weights = vit(images.float().to(device))
                loss = self.criterion(outputs.float(), tx.reshape(outputs.shape).float().to(device))
                loss.backward()

                # Apply perturbation to weights to knock out of saddle points
                is_perturb = False
                if epoch % 10 == 0:
                    
                    if config.PERTURBATION > 0. and len(train_losses) > 1:

                        last_k_diff = abs(np.max(train_losses[-last_k_perturb:]) -
                                          np.min(train_losses[-last_k_perturb:]))

                        # If the last 30 epochs do not differ by the threshold and
                        # we're above the loss limit, we may be in a saddle point.
                        # Perturb the weights of the model
                        if ((last_k_diff < loss_threshold) and
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

                perturb_str = "* Saddle conditions met! Perturbing model weights *" if is_perturb else ""
                print_fl('[%d] train loss: %.5f, validation loss %.5f, %s %s' %
                      (epoch + 1, train_loss, validation_loss, timer.get_time(), perturb_str))

                torch.save(vit.state_dict(), model_path)

                # Save intermediate model
                if epoch % 2000 == 0 :
                   torch.save(vit.state_dict(), f"{model_path}.{epoch}")

                epochs_arr.append(epoch)
                validation_losses.append(validation_loss)
                train_losses.append(train_loss)

                loss_df = pd.DataFrame({
                        'epoch': epochs_arr,
                        'train_loss': train_losses,
                        'validation_loss': validation_losses
                    })

                loss_df.to_csv(loss_path, index=False)

                # Plot loss
                fig = plot_loss_progress(loss_df, m=50)
                plt.savefig(loss_fig_path, dpi=150)
                plt.close(fig)
                plt.cla()
                plt.clf()

                if epoch % 100 == 0 :

                    # Plot accuracy
                    fig, train_r2, valid_r2, test_r2 = plot_predictions(vit, vit_data, trainloader, validationloader, 
                        testloader, 512, device=device)
                    plt.savefig(f'{out_dir}/predictions.png', dpi=150)
                    plt.close(fig)
                    plt.cla()
                    plt.clf()

            # End early if loss is reached
            if train_loss < self.stoploss_value:
                print_fl(f"[{epoch}] Stop loss reached: {train_loss} < {self.stoploss_value}. Ending early.")
                break

        print_fl(f'Finished Training {timer.get_time()}')

    def compute_validation_loss(self, dataloader, num_batches=-1):

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


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    return device


def load_model_config(config):
    model_class = ViT

    vit = model_class(config)
    return vit


def load_model_dir(model_dir):
    config_path = f"{model_dir.replace('/', '.')}.config"
    config = importlib.import_module(config_path)
    vit = load_model_config(config)
    vit.load_state_dict(torch.load(f"{model_dir}/model.torch", map_location=torch.device('cpu')))
    return vit, config


def main():

    # Load config from command-line
    config_name = sys.argv[1]
    config = importlib.import_module(f"config.{config_name}")
    model_path = config_name

    # Load ViT model from config
    vit = load_model_config(config)
    print_fl(f"Config: {vit.config_repr()}")

    # Data loading
    print_fl("Loading data...")
    dataset = load_cd_data()
    dataloader = ViTDataLoader(dataset)
    print_fl(f"Dataloader split: {dataloader.split_repr()}")

    # Initialize trainer
    print_fl("Initializing trainer...")
    trainer = ViTTrainer(vit, dataloader)
    trainer.setup()
    print_fl(f"Writing to {trainer.out_dir}")

    # Train
    print_fl("Training...")
    trainer.train() 

if __name__ == '__main__':
    main()
