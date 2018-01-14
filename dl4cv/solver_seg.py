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
                 loss_func=torch.nn.CrossEntropyLoss(ignore_index=-1)):
        
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
        x = train_loader.dataset[:][0] # 1st element contains the data. 3x240x240  (Torch Tensor)
        y = train_loader.dataset[:][1]  # 240x240 tensor
        x_val = val_loader.dataset[:][0]  # 1st element contains the data. 3x240x240  (Torch Tensor)
        y_val = val_loader.dataset[:][1] # 240x240 tensor
        '''
        
        
        
        #num_train = len(train_loader.dataset)
        #iters_per_epoch = max(num_train // train_loader.batch_size, 1) # epoch is like coverring the whole dataset once
        num_iterations = num_epochs * iter_per_epoch
        it = 1
        
        optim = self.optim([{'params': model.upsample.parameters()}], **self.optim_args)
        
        for epoch in range(num_epochs):
            #y = next(train_loader.sampler.__iter__()) I don't know why such a sampling would be used
            #indx = np.random.choice(num_train, train_loader.batch_size,  replace= True)
            
            #indx = torch.from_numpy(indx)
            '''
            x_batch = x.numpy()[indx] 
            x_batch = torch.from_numpy(x_batch)
            y_batch = y.numpy()[indx]
            y_batch = torch.from_numpy(y_batch)
            '''
            
            for data in train_loader:
                x_batch, y_batch = data
                x_batch, y_batch = Variable(x_batch), Variable(y_batch)
              
                #y_batch.type(torch.FloatTensor)  This was required for MSE() loss function
                if model.is_cuda:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()


                output = model.forward(x_batch)  # it returns Variable. Access data by output.data
                #output = model(Variable(x_batch))  # It also returns Variable
                #output = output.cpu()
                #print(output.data)
                #output = output.permute(0,2,3,1).contiguous()
                #output = output.view(-1, output.size()[-1])     
                #y_batch = y_batch.view(-1)
                #y_batch[y_batch == -1] = 23

     
                loss = self.loss_func(output, y_batch)
                


                ######  Updating weights   #########
                # create an optimizer
                #print(list(model.upsample.parameters()))
                #optim = self.optim(model.parameters(), **self.optim_args)
                
                
                
                optim.zero_grad()   # zero the gradient buffers
                loss.backward()
                optim.step()    # Does the update based on the accumulated gradients
                #print(list(model.upsample.parameters()))
                
                if ((log_nth !=0) and (it%log_nth ==0)):
                    self.train_loss_history.append(loss)
                  
                #if it % 100 == 0:
                #    print('iteration %d / %d: loss %f' % (it, num_iterations, loss))
                    
                
                targets_mask = y_batch >= 0
                _, preds = torch.max(output, 1)
                train_acc = np.mean((preds == y_batch)[targets_mask].data.cpu().numpy())
                
                #train_acc = self.predict(output, y_batch)
                #print('train acc %f',train_acc)
                it = it+1

            # Every epoch, check train and val accuracy and decay learning rate.
            
            # Check accuracy
            #print("I am about to train prediction")
            #output = output.cpu()
            #train_acc = self.predict(output, y_batch)
            print('train acc %f',train_acc)
            self.train_acc_history.append(train_acc)
            
            val_acc = 0
            for data in val_loader:
                x_val, y_val = data
                x_val, y_val = Variable(x_val), Variable(y_val)
                if model.is_cuda:
                    x_val, y_val = x_val.cuda(), y_val.cuda()

                output_val = model.forward(x_val)
                '''
                output_val = output_val.permute(0,2,3,1).contiguous()
                output_val = output_val.view(-1, output_val.size()[-1])     
                y_val = y_val.view(-1)
                
                # removing unmapped pixel
                output_val = output_val.cpu()
                output_val_np = output_val.data.numpy()
                y_val_np = y_val.numpy()
                unmap_pix = y_val_np>=0
                indx_pix = np.where(unmap_pix ==1)[0]
                output_val = Variable(torch.from_numpy(output_val_np[indx_pix,]))
                y_val = torch.from_numpy(y_val_np[indx_pix])
                #print(y_batch)
                #print("I am about to val prediction")
                '''
                
                _, preds = torch.max(output_val, 1)
                targets_mask =y_val >= 0
                val_acc += np.mean((preds == y_val)[targets_mask].data.cpu().numpy())
                #val_acc = val_acc + self.predict(output_val.cpu(),y_val)

            val_acc = val_acc/len(val_loader)  #avg
            print('val acc %f', val_acc)
            self.val_acc_history.append(val_acc)
            
        
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

        
    def predict(self, output, labels):
            # removing unmapped pixel
            #print(output)
            
            output = output.permute(0,2,3,1).contiguous()
            output = output.view(-1, output.size()[-1])     
            labels = labels.view(-1)
            
            #print(labels)
            output_np = output.data.numpy()
            labels_np = labels.numpy()
            unmap_pix = labels_np<23
            indx_pix = np.where(unmap_pix ==1)[0]
            output = output_np[indx_pix,]
            labels = labels_np[indx_pix]
            #print(output)
            y_pred = np.argmax(output, axis=1)
            acc = np.mean((y_pred == labels ))
 
            
            
            return acc
        
        

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