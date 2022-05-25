
import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import numpy as np
import importlib
import shutil
import uuid

from datetime import date
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from scipy.stats import spearmanr
from src.vit_data_2 import ViTData, load_cd_data
from src.timer import Timer
from src.utils import print_fl, mkdir_safe
from src.vit_old import ViT
from src.vit_train import plot_loss_progress


def load_data():
    
    vit_data = load_cd_data()

    # Test set is time point 120 minutes
    indices = np.arange(len(vit_data))
    select_120 = vit_data.times == 120.0
    test_indices = indices[select_120]
    train_validation_indices = indices[~select_120]

    # 90%/10% training, validation set
    num_training = int(0.9 * len(train_validation_indices))
    train_indices = np.array(sorted(np.random.choice(train_validation_indices, size=num_training, replace=False)))
    validation_indices = np.setdiff1d(train_validation_indices, train_indices)

    trainset = torch.utils.data.Subset(vit_data, train_indices)
    validationset = torch.utils.data.Subset(vit_data, validation_indices)
    testset = torch.utils.data.Subset(vit_data, test_indices)

    return vit_data, trainset, validationset, testset


def load_dataloaders(vit_data, trainset, validationset, testset,
        batch_size):

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    return trainloader, validationloader, testloader


def compute_validation_loss(vit, testloader, criterion, 
    device=torch.device('cpu'), num_batches=-1):
    
    with torch.no_grad():

        i = 0
        running_loss = 0
        for data in testloader:
            images, tx, _, _, _ = data
            images = images.to(device)
            tx = tx.to(device)

            # calculate outputs by running images through the network
            outputs, weights = vit(images)
            outputs = outputs.to(device)
            loss = criterion(outputs, tx.reshape(outputs.shape))
            running_loss += loss.item()

            i+=1
            if i == num_batches: break

    return running_loss / i


def load_model_config(config):

    vit = ViT(in_channels=config.IN_CHANNELS, 
              img_size=config.IMG_SIZE,
              patch_size=config.PATCH_SIZE, 
              emb_size=config.EMB_SIZE,
              num_heads=config.NUM_HEADS,
              transformer_depth=config.DEPTH,
              forward_expansion=config.FORWARD_EXPANSION,
              n_classes=1, 
              att_drop_p=config.DROPOUT,
              forward_drop_p=config.DROPOUT)
    vit.config = config
    return vit


def load_model_dir(model_dir):
    config_path = f"{model_dir.replace('/', '.')}.config"
    config = importlib.import_module(config_path)
    vit = load_model_config(config)
    vit.load_state_dict(torch.load(f"{model_dir}/model.torch", map_location=torch.device('cpu')))
    return vit, config


def main():

    args = ['', 'resume', 'output/complex_test']

    if args[1] == 'resume':
        resume = True
        resume_path = args[2]
    
    out_dir = resume_path
    vit, config = load_model_dir(resume_path)

    model_path = f'{resume_path}/model.torch'
    vit.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    loss_path = f"{resume_path}/loss.csv"
    loss_fig_path = f"{resume_path}/loss.png"

    if torch.cuda.is_available():
        device = torch.device("cuda") 
        print_fl(f"Using {device} {torch.cuda.device_count()} " \
                 f"GPU devices. e.g. {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print_fl(f"Using {device}")

    vit = vit.to(device)

    timer = Timer()

    # Which time to use as first channl
    if config.IN_CHANNELS == 2:
        channel_1_time = config.CHANNEL_1
        print_fl(f"Using channel {channel_1_time}")
    elif config.CHANNEL_1 is not None:
        raise ValueError(f"Invalid channel config, {in_channels} channels, with channel 1: {config.CHANNEL_1}")
    else:
        channel_1_time = None

    if not resume:
        mkdir_safe(out_dir)
        shutil.copyfile(f"config/{config_name}.py", f"{out_dir}/config.py")

    np.random.seed(123)

    (vit_data, trainset, validationset, 
        testset) = load_data()

    print_fl(f"Using model {model_path}")
    print_fl(f"Using output directory {out_dir}")
    print_fl(f"Training: {len(trainset)}\nValidation: "
             f"{len(validationset)}\nTesting: {len(testset)}")

    (trainloader, 
     validationloader, testloader) = load_dataloaders(vit_data, trainset, validationset, testset,
        batch_size=config.BATCH_SIZE)

    # How many loss values to check for in train_losses array to detect saddle points, losses are
    # collected every 10 epochs. And threshold considered as "stuck" in saddle point.
    # perturbation settings if enabled
    last_k_perturb = 5
    loss_threshold = 1e-3

    # Value of training loss to stop training
    stoploss_value = 1e-5

    epochs = 100000

    optimizer_name = config.OPTIMIZER
    print_fl(f"Using optimizer {optimizer_name}")

    criterion = nn.MSELoss()

    print_fl(f"Using Perturbation SGD with perturb={config.PERTURBATION} "
             f"until < {config.PERTURBATION_LOSS_LIM}")

    if optimizer_name == 'SGD':

        lr = 0.001
        momentum = 0.9

        print_fl(f"Learning rate={lr}, momentum={momentum}")
        optimizer = optim.SGD(vit.parameters(), lr=lr, momentum=momentum)

    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(vit.parameters(), lr=0.001)

    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(vit.parameters(), lr=0.01)

    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(vit.parameters(), lr=1.0, rho=0.9)

    else:
        ValueError(f"Invalid optimizer: {optimizer_name}")
    
    num_batches = len(trainloader)

    if resume:
        losses_df = pd.read_csv(f"{resume_path}/loss.csv")
        epochs_arr = list(losses_df['epoch'].values)
        train_losses = list(losses_df['train_loss'].values)
        validation_losses = list(losses_df['validation_loss'].values)
        last_epoch = epochs_arr[-1]
        print_fl(f"Resuming from {last_epoch} with {num_batches} "
                 f"batches of size {config.BATCH_SIZE}...")
    else:
        epochs_arr = []
        validation_losses = []
        train_losses = []
        last_epoch = 0

        print_fl(f"Starting training, {epochs} epochs with {num_batches} "
                 f"batches of size {config.BATCH_SIZE}...")

    epochs_to_run = np.arange(last_epoch+1, last_epoch+epochs)

    for epoch in epochs_to_run:

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            images, tx, _, _, _ = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, weights = vit(images.float().to(device))
            loss = criterion(outputs, tx.float().reshape(outputs.shape).to(device))
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

            validation_loss = compute_validation_loss(vit, validationloader,
                criterion, device)

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


        # End early if loss is reached
        if train_loss < stoploss_value:
            print_fl(f"[{epoch}] Stop loss reached: {train_loss} < {stoploss_value}. Ending early.")
            break


    print_fl(f'Finished Training {timer.get_time()}')


if __name__ == '__main__':
    main()
