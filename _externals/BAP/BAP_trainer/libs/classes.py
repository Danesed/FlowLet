import os
import sys
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,StratifiedKFold

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts


''' Classes for training, validation, and testing of models in PyTorch.'''


class reg_dataset(torch.utils.data.Dataset):

    
    def __init__(self,
                dataset_path: str,
                data: pd.DataFrame,
                _min: float,
                _max: float):
        self.dataset_path = dataset_path;
        self.data = data; 
        self.inputs_dtype = torch.float32;
        self.min = _min;
        self.max = _max;
        
    def __len__(self):
        return self.data.shape[0];


    def __getitem__(self, index: int):

        ID = str(self.data.iloc[index]['id']);
        target = float(self.data.iloc[index]['age']);
        input_path = f"{self.dataset_path}/{ID}";
        input = np.load(input_path);
        input = (input - self.min)/(self.max - self.min);
        input = np.clip(input, 0.0, 1.0)     
        input = np.squeeze(input, axis = 0);
        input = torch.from_numpy(input).type(self.inputs_dtype);
        return input, target, ID;
  
  

class trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 no_cuda: bool = True,
                 epochs: int = 100,
                 epoch: int = 0,
                 patience: int = 10,
                 delta: float = 0.0,
                 cp_path: str = "checkpoint",
                 check: int = None,
                 ):

        
        self.criterion = criterion;
        self.optimizer = optimizer;
        self.lr_scheduler = lr_scheduler;
        self.training_DataLoader = training_DataLoader;
        self.validation_DataLoader = validation_DataLoader;
        self.device = device;
        self.model = model.to(self.device);
        self.no_cuda = no_cuda;
        self.epochs = epochs;
        self.epoch = epoch;
        self.patience = patience;
        self.cp_path = cp_path;
        self.delta = delta;
        self.check = check;ß
        self.training_loss_mean = [];
        self.training_loss_std = [];
        self.validation_loss_mean = [];
        self.validation_loss_std = [];
        self.learning_rate = [];

    
    def run_trainer(self):
        early_stopping = EarlyStopping(patience=self.patience,
                                       verbose = False,
                                       delta = self.delta,
                                       path = f"{self.cp_path}_eStop_{self.epoch}.pt");
        stop = False;
        loss_pre = 0.0;

        while self.epoch < self.epochs and stop == False:
            self.epoch += 1;
            print(f"\nTRAINING: Epoch {self.epoch}/{self.epochs} [Previous loss: {loss_pre:.4f}, LR: {self.optimizer.param_groups[0]['lr']}]")
            self._train();

            
            if not self.no_cuda:
              torch.cuda.empty_cache();

            
            if self.validation_DataLoader is not None:
                print("\nVALIDATION");
                self._validate();
                if not self.no_cuda:
                      torch.cuda.empty_cache();
                loss_pre = self.validation_loss_mean[-1];
                early_stopping(loss_pre, self.model, self.epoch);

                if early_stopping.early_stop:
                      stop = True;
            else:
                loss_pre = self.training_loss_mean[-1];

            if self.lr_scheduler is not None:
                self.lr_scheduler.step();
            
            if (self.check is not None) and (self.epoch % self.check == 0):
              torch.save(self.model.state_dict(), f"{self.cp_path}_checkpoint_epoch_{self.epoch}.pt");


        if stop == True:
          print(f"\n Training stopped (early stopping in epoch {self.epoch})");
        else:
          print("\n Training complete.");

        if self.cp_path is not None:
          torch.save(self.model.state_dict(), f"{self.cp_path}_COMPLETED_epoch_{self.epoch}.pt");

        return self.model, self.training_loss_mean, self.training_loss_std,self.validation_loss_mean, self.validation_loss_std,self.learning_rate


    
    def _train(self):
        self.model.train();
        train_losses = list();
        tot = (self.training_DataLoader.dataset.__len__());
        counter = 0;
        
        for x,y,_ in self.training_DataLoader:
            counter = counter + x.shape[0];
            sys.stdout.write(f"\r{counter}/{tot} ");
            input, target = x.to(self.device), y.to(self.device);
            self.optimizer.zero_grad();
            pred = self.model(input);
            pred = torch.reshape(pred,target.shape);
            loss = self.criterion(pred, target);
            loss_value = loss.tolist();
            train_losses.extend(loss_value);
            loss = torch.mean(loss);
            loss.backward();
            self.optimizer.step();
            del input,target;
            sys.stdout.flush();

        self.training_loss_mean.append(np.mean(train_losses));
        self.training_loss_std.append(np.std(train_losses));
        self.learning_rate.append(self.optimizer.param_groups[0]['lr']);


    
    def _validate(self):

        self.model.eval()
        valid_losses = []
        tot = (self.validation_DataLoader.dataset.__len__());
        counter = 0;

        
        for x,y,_ in self.validation_DataLoader:
            counter = counter + x.shape[0];
            sys.stdout.write(f"\r{counter}/{tot} ");
            input, target = x.to(self.device), y.to(self.device);
            
            with torch.no_grad():
                pred = self.model(input);
                pred = torch.reshape(pred,target.shape);
                loss = self.criterion(pred, target)
                loss_value = loss.tolist();
                valid_losses = valid_losses + loss_value;

            del input, target;
            sys.stdout.flush();

        self.validation_loss_mean.append(np.mean(valid_losses));
        self.validation_loss_std.append(np.std(valid_losses));              
        


class tester:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 test_DataLoader: torch.utils.data.Dataset,
                 criterion: torch.nn.Module,
                 no_cuda: bool,
                 ):
        self.test_DataLoader = test_DataLoader;
        self.device = device;
        self.model = model.to(self.device);
        self.criterion = criterion;
        self.no_cuda = no_cuda;
        self.test_losses = [];
        self.ID_list = [];
        self.target_list = [];
        self.preds = [];


    
    def run_tester(self):
        self.model.eval()
        tot = (self.test_DataLoader.dataset.__len__());
        counter = 0;
        
        for x,y,id in self.test_DataLoader:
            counter = counter + x.shape[0];
            sys.stdout.write(f"\r{counter}/{tot}");
            
            input, target = x.to(self.device), y.to(self.device);
            with torch.no_grad():
                pred = self.model(input)
                pred = torch.reshape(pred,target.shape)
                loss = self.criterion(pred, target);
                loss_value = loss.tolist();
                self.test_losses = self.test_losses + loss_value;
                self.target_list = self.target_list + (y.tolist());
                self.preds = self.preds + (pred.tolist());
                self.ID_list = self.ID_list + (list(id));

            del input, target;
            sys.stdout.flush();

            if not self.no_cuda:
              torch.cuda.empty_cache();
        return self.test_losses, self.ID_list, self.target_list, self.preds;
  
  
  

class EarlyStopping:

    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pt', trace_func=print, delay = 30):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
        self.delay = delay;
        print(f"Early stop (delay: {self.delay})");

    def __call__(self, val_loss, model, epoch):
        
        if epoch == self.delay:
          print('Early stopping on!')

        if epoch > self.delay:
          score = -val_loss
          
          if self.best_score is None:
              self.best_score = score
              self.save_checkpoint(val_loss, model)
          elif score < self.best_score + self.delta:
              self.counter += 1
              if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
              if self.counter >= self.patience:
                  self.early_stop = True
          else:
              self.best_score = score
              self.save_checkpoint(val_loss, model)
              self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        self.val_loss_min = val_loss
        
        
        