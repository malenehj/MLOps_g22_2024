"""
Simple baseline modle which always will guess on the most fequient class 
of the traning data.

author: Mathias Nissen
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class BaselineModel(nn.Module):

    def __init__(self, traning_data: DataLoader):

        # Creating a numpy array containing all classes of the training data:
        data_iter = iter(traning_data)
        all_train_labels = np.array([])
        for data in data_iter:
            all_train_labels = np.hstack((all_train_labels, data[-1]))

        # Getting all class labels, and their number of occurrences:
        labels, occurrences = np.unique(all_train_labels, return_counts=True)
        # The maximum number of occurrences of one class:
        max_occur = np.max(occurrences)
        
        # Getting most frequent class (emotion):
        index = np.where(max_occur == occurrences)[0][0] # Index of most frequent class
        max_class = labels[index] # Most frequent class

        # Defining the guess which the baseline will use:
        self.guess = torch.tensor([max_class])
        
    def forward(self, data: torch.tensor = torch.tensor([])):

        # Determining the shape of the input (batch or single data point).
        # The function will return an output with the same number of data points
        # (the same batch size).
        if len(data.shape) == 1:
            num_data = 1
        elif len(data.shape) == 2:
            num_data = len(data)
        else:
            raise Exception("Unrecognized data shape") 
        
        # Generating output of appropriate shape, of the guess determined in init:
        output = np.ones(num_data) * self.guess.numpy()
        output = torch.tensor(output)
        return output