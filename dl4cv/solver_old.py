from random import shuffle
import numpy as np
#import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.MSELoss()):
        
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_data, train_labels, val_data, val_labels, batch_size =50, log_nth=0, num_epochs=10):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        
        self._reset_histories()
        #iter_per_epoch = len(train_loader)  This was here from the beginning. I think it didn't have any importance

        if torch.cuda.is_available():
            model.cuda()
        
        print('START TRAIN.')
        ########################################################################
     
        ########################################################################
        
        x = train_data # 1st element contains the data. 3x32x32  (Torch Tensor)
        y = train_labels  # 2nd element contains labels  (Simple Variable)
        
        x_val = val_data  # Torch tensor
        y_val = val_labels # simple variable
      
        
        
        num_train = len(train_data)
        iters_per_epoch = max(num_train // batch_size, 1) # epoch is like coverring the whole dataset once
        num_iterations = num_epochs * iters_per_epoch
           
            
        for it in range(num_iterations):
            #y = next(train_loader.sampler.__iter__()) I don't know why such a sampling would be used
            indx = np.random.choice(num_train, batch_size,  replace= True)
            
            indx = torch.from_numpy(indx)
            
            x_batch = x[indx]   
            y_batch = y[indx]
            y_batch = y_batch.view(-1, self.num_flat_features(y_batch)) 
          
            #y_batch.type(torch.FloatTensor)  This was required for MSE() loss function
       
            
            output = model.forward(Variable(x_batch))  # it returns Variable. Access data by output.data
            #output = model(Variable(x_batch))  # It also returns Variable
 
            loss = self.loss_func(output, Variable(y_batch))
            
   
            ######  Updating weights   #########
            # create an optimizer
            optim = self.optim(model.parameters(), **self.optim_args)
            optim.zero_grad()   # zero the gradient buffers
            loss.backward()
            optim.step()    # Does the update based on the accumulated gradients
  
            if ((log_nth !=0) and (it%log_nth ==0)):
                self.train_loss_history.append(loss.data.numpy())
       
         
            if it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iterations, loss.data.numpy()))

           
        
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

 
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
