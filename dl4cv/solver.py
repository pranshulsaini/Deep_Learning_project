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
                         "weight_decay": 5e-4}

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

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
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
        iter_per_epoch = len(train_loader)  #This was here from the beginning. I think it didn't have any importance

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could look something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        '''
        x = train_loader.dataset[:][0] # 1st element contains the data. 3x32x32  (Torch Tensor)
        y = train_loader.dataset[:][1]  # 2nd element contains labels  (Simple Variable)
        y = torch.from_numpy(y) # I am being consistent in dealing with tensors
        x_val = val_loader.dataset[:][0]  # Torch tensor
        y_val = val_loader.dataset[:][1] # simple variable
        y_val = torch.from_numpy(y_val) # I am being consistent in dealing with tensors
        '''
        
        #num_train = len(train_loader.dataset)
        #iters_per_epoch = max(num_train // train_loader.batch_size, 1) # epoch is like coverring the whole dataset once
        num_iterations = num_epochs * iter_per_epoch
       
        it = 1
        for epoch in range(num_epochs):
            #y = next(train_loader.sampler.__iter__()) I don't know why such a sampling would be used
            '''
            indx = np.random.choice(num_train, train_loader.batch_size,  replace= True)
            
            indx = torch.from_numpy(indx)
            
            x_batch = x[indx]   
            y_batch = y[indx]
            '''
            #y_batch.type(torch.FloatTensor)  This was required for MSE() loss function
       
            for data in train_loader:
                x_batch, y_batch = data
            
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
                
                it = it +1

            # Every epoch, check train and val accuracy and decay learning rate.
            
            # Check accuracy
            train_acc = (self.predict(output.data) == y_batch).sum()/len(self.predict(output.data) == y_batch)
            self.train_acc_history.append(train_acc)
            
            val_acc = 0
            for data in val_loader:
                x_val, y_val = data
                output_val = model.forward(Variable(x_val)) 
                val_acc = val_acc + (self.predict(output_val.data) == y_val).sum()/len(self.predict(output_val.data) == y_val)

            val_acc = val_acc/len(val_loader)  #avg
            self.val_acc_history.append(val_acc)
        
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

        
    def predict(self, X):
        
      
            X = X.numpy() # converting into numpy
            y_pred = np.argmax(X, axis=1)
            return torch.from_numpy(y_pred)

def my_loss(x, y):  # not used
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    x = x.data
    
    y_mat = torch.zeros((x.shape[0],x.shape[1])) # dimensions N x C
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y_mat[i,y.type(torch.ByteTensor)[j]] = 1
    
    max_values = torch.max(x, 1)[0]   # 2nd index would give the indices
    max_values= max_values.unsqueeze(1).repeat(1,x.shape[1])
    
    probs = torch.exp(x - max_values)
    probs_sum = torch.sum(probs,1)
    probs_sum = probs_sum.unsqueeze(1).repeat(1,x.shape[1])
    
    final_probs = probs/probs_sum
   
    criterion = nn.MSELoss()
    loss = criterion(final_probs, y_mat)
    return loss