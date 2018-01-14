"""ClassificationCNN"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ClassificationCNN(nn.Module):

    def __init__(self):
        """
        Class constructor which preinitializes NN layers with trainable
        parameters.
        """
        super(ClassificationCNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # conv kernel
        self.conv1 = nn.Conv2d(5, 6, (1,2))
        self.conv2 = nn.Conv2d(6, 4, (1,3))
        #self.conv1.weight.data = self.conv1.weight.data * 2 # printing the automatically initialised weights
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28, 20)
        

    def forward(self, x):
        """
        Forwards the input x through each of the NN layers and outputs the result.
        """
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))  # before choosing the max one, they are performing ReLU
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))  # 7x4
        
        # An efficient transition from spatial conv layers to flat 1D fully 
        # connected layers is achieved by only changing the "view" on the
        # underlying data and memory structure.
        
        x = x.view(-1, self.num_flat_features(x)) # the shape of x changes from [1,16,5,5] to [1,400]
        x = F.relu(self.fc1(x))   # this step is multiplying the weight matrix defined in self.fc1 with x of shape[1,400]
        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
