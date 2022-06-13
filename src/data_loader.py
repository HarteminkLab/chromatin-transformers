
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import torch


class ViTDataLoader:

    def __init__(self, dataset, batch_size=64, split_type="chrom", split_arg=4, valid_type="proportion", valid_arg=0.1):

        # Load data
        self.dataset = dataset

        self.split_type = split_type
        self.split_arg = split_arg

        np.random.seed(123)

        if split_type == 'hybrid':
            self.trainset, self.validationset, self.testset = testtrain_split_hybrid(self.dataset, test_prop=split_arg, 
                valid_type=valid_type, valid_arg=valid_arg)
        elif split_type == 'proportion':
            self.trainset, self.validationset, self.testset = testtrain_split_prop(self.dataset, test_prop=split_arg, 
                valid_type=valid_type, valid_arg=valid_arg)
        elif split_type == 'chrom':
            self.trainset, self.validationset, self.testset = testtrain_split_chrom(self.dataset, test_chrom=split_arg, 
                valid_type=valid_type, valid_arg=valid_arg)
        elif split_type == 'time':
            self.trainset, self.validationset, self.testset = testtrain_split_time(self.dataset, test_time=split_arg, 
                valid_type=valid_type, valid_arg=valid_arg)
        else:
            raise ValueError("Invalid split type")

        # Load train loader
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size)
        self.validationloader = torch.utils.data.DataLoader(self.validationset, batch_size=batch_size)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size)

    def split_repr(self):
        return (f"Split: {self.split_type},{self.split_arg}; Training: {len(self.trainset)}; "
                f"Validation: {len(self.validationset)}; Testing: {len(self.testset)}")

    def save_indices(self, save_path):
        test_indices = self.testloader.dataset.indices
        train_indices = self.trainloader.dataset.indices
        valid_indices = self.validationloader.dataset.indices

        test_df = pd.DataFrame({'index': test_indices, 'set': 'test'})
        train_df = pd.DataFrame({'index': train_indices, 'set': 'train'})
        valid_df = pd.DataFrame({'index': valid_indices, 'set': 'validation'})
        indices_df = pd.concat([test_df, train_df, valid_df]).reset_index(drop=True)
        indices_df.to_csv(save_path)

def get_train_valid_indices(dataset, test_indices, trainvalid_indices, valid_type, valid_arg):

    n = len(dataset)

    # Randomly subset training set for validation
    if valid_type == 'proportion' or valid_type == 'hybrid':
        validation_indices, train_indices = split_orfs_subset(dataset, valid_arg, trainvalid_indices)
    # Subset training set by chromosome
    elif valid_type == 'chrom':
        validation_indices = np.arange(n)[dataset.chrs == valid_arg]
        train_indices = np.arange(n)[dataset.chrs != valid_arg]
        train_indices = np.intersect1d(trainvalid_indices, train_indices)
    # Subset training set by time
    elif valid_type == 'time':
        validation_indices = np.arange(n)[dataset.times == valid_arg]
        train_indices = np.arange(n)[dataset.times != valid_arg]
        train_indices = np.intersect1d(trainvalid_indices, train_indices)
    else:
        raise ValueError(f"Unsupported validation {valid_type}")

    return train_indices, validation_indices


def testtrain_split_hybrid(dataset, test_prop=0.20, valid_type="hybrid", valid_arg=0.1):
    """
    Split the dataset into train, validation, and test datasets based on proportion of randomly selected ORFs
    """

    n = len(dataset)
    all_indices = np.arange(n)

    # Preserve timepoint zero for training
    indices_zero = all_indices[dataset.times == 0]
    indices_non_zero = all_indices[dataset.times != 0]

    test_indices, trainvalid_indices = split_orfs_subset(dataset, test_prop, indices_non_zero)
    train_indices, validation_indices = get_train_valid_indices(dataset, test_indices, trainvalid_indices, 
        valid_type, valid_arg)

    train_indices = np.concatenate([indices_zero, train_indices])

    trainset = torch.utils.data.Subset(dataset, train_indices)
    validationset = torch.utils.data.Subset(dataset, validation_indices)
    testset = torch.utils.data.Subset(dataset, test_indices)

    return trainset, validationset, testset


def testtrain_split_prop(dataset, test_prop=0.20, valid_type="proportion", valid_arg=0.1):
    """
    Split the dataset into train, validation, and test datasets based on proportion of randomly selected ORFs
    """

    n = len(dataset)
    all_indices = np.arange(n)
    test_indices, trainvalid_indices = split_orfs_subset(dataset, test_prop, all_indices)
    train_indices, validation_indices = get_train_valid_indices(dataset, test_indices, trainvalid_indices, 
        valid_type, valid_arg)

    trainset = torch.utils.data.Subset(dataset, train_indices)
    validationset = torch.utils.data.Subset(dataset, validation_indices)
    testset = torch.utils.data.Subset(dataset, test_indices)

    return trainset, validationset, testset


def split_orfs_subset(dataset, proportion_1, indices_set):
    """
    Split a dataset by proportion on set of ORFs. Randomly select based on ORFs. Use indices_set if
    subsetting entire dataset.
    """

    # Select the ORFs available to subselect
    orfs_list = dataset.orfs[indices_set]
    orfs_set = np.array(list(set(orfs_list)))
    m = len(orfs_set)

    num_split_1 = int(proportion_1 * m)
    set_1_orfs = np.random.choice(orfs_set, size=num_split_1, replace=False)

    set_1_indices = indices_set[[o in set_1_orfs for o in orfs_list]]
    set_2_indices = indices_set[[o not in set_1_orfs for o in orfs_list]]

    return set_1_indices, set_2_indices


def testtrain_split_time(dataset, test_time=120, valid_type="proportion", valid_arg=0.1):
    """
    Split the dataset into train, validation, and test datasets
    """

    n = len(dataset)

    test_indices = np.arange(n)[dataset.times == test_time]
    trainvalid_indices = np.arange(n)[dataset.times != test_time]

    train_indices, validation_indices = get_train_valid_indices(dataset, test_indices, trainvalid_indices, 
        valid_type, valid_arg)

    trainset = torch.utils.data.Subset(dataset, train_indices)
    validationset = torch.utils.data.Subset(dataset, validation_indices)
    testset = torch.utils.data.Subset(dataset, test_indices)

    return trainset, validationset, testset


def testtrain_split_chrom(dataset, test_chrom=4, valid_type="proportion", valid_arg=0.1):
    """
    Split the dataset into train, validation, and test datasets
    """

    n = len(dataset)

    test_indices = np.arange(n)[dataset.chrs == test_chrom]
    trainvalid_indices = np.arange(n)[dataset.chrs != test_chrom]

    train_indices, validation_indices = get_train_valid_indices(dataset, test_indices, trainvalid_indices, 
        valid_type, valid_arg)

    trainset = torch.utils.data.Subset(dataset, train_indices)
    validationset = torch.utils.data.Subset(dataset, validation_indices)
    testset = torch.utils.data.Subset(dataset, test_indices)

    return trainset, validationset, testset

