
import sys
sys.path.append('.')

import numpy as np
import torch


class ViTDataLoader:

    def __init__(self, dataset, batch_size=64, split_type="chrom", split_arg=4):

        # Load data
        self.dataset = dataset

        self.split_type = split_type
        self.split_arg = split_arg

        if split_type == 'chrom':
            self.trainset, self.validationset, self.testset = testtrain_split_chrom(self.dataset, test_chrom=split_arg)
        elif split_type == 'time':
            self.trainset, self.validationset, self.testset = testtrain_split_time(self.dataset, test_time=split_arg)
        else:
            raise ValueError("Invalid split type")

        # Load train loader
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size)
        self.validationloader = torch.utils.data.DataLoader(self.validationset, batch_size=batch_size)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size)

    def split_repr(self):
        return (f"Split: {self.split_type},{self.split_arg}; Training: {len(self.trainset)}; "
                f"Validation: {len(self.validationset)}; Testing: {len(self.testset)}")


def testtrain_split_time(dataset, train_validation_split=0.9, test_time=120):
    """
    Split the dataset into train, validation, and test datasets
    """

    n = len(dataset)

    test_indices = np.arange(n)[dataset.times == test_time]
    trainvalidation_idx = np.arange(n)[dataset.times != test_time]

    # 90%/10% training, validation set
    num_training = int(train_validation_split * len(trainvalidation_idx))
    train_indices = np.array(sorted(np.random.choice(trainvalidation_idx, size=num_training, replace=False)))
    validation_indices = np.setdiff1d(trainvalidation_idx, train_indices)

    trainset = torch.utils.data.Subset(dataset, train_indices)
    validationset = torch.utils.data.Subset(dataset, validation_indices)
    testset = torch.utils.data.Subset(dataset, test_indices)

    return trainset, validationset, testset


def testtrain_split_chrom(dataset, train_validation_split=0.9, test_chrom=4):
    """
    Split the dataset into train, validation, and test datasets
    """

    n = len(dataset)

    test_indices = np.arange(n)[dataset.chrs == test_chrom]
    trainvalidation_idx = np.arange(n)[dataset.chrs != test_chrom]

    # 90%/10% training, validation set
    num_training = int(train_validation_split * len(trainvalidation_idx))
    train_indices = np.array(sorted(np.random.choice(trainvalidation_idx, size=num_training, replace=False)))
    validation_indices = np.setdiff1d(trainvalidation_idx, train_indices)

    trainset = torch.utils.data.Subset(dataset, train_indices)
    validationset = torch.utils.data.Subset(dataset, validation_indices)
    testset = torch.utils.data.Subset(dataset, test_indices)

    return trainset, validationset, testset

