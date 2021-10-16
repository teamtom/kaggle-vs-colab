import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, SubsetRandomSampler, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint

import wandb

import seaborn as sns


class Pipeline:

   def __init__(self, task, optimizer, loss_fn=None, random_seed=1, batch_size=32, learning_rate=0.001, epochs=10, lr_sch=None, lr_sch_args={},
      wandb_log=False, progressbar=False):
      self.task = task # regression | classification
      self.random_seed = random_seed
      self.batch_size = batch_size
      self.learning_rate = learning_rate

      self.epochs = epochs

      self.loss_fn = loss_fn
      self.optim = optimizer

      self.threshold = 0.5

      self.class_weight = None

      self.lr_sch = lr_sch
      self.lr_sch_args = lr_sch_args

      self.progressbar = progressbar
      self.wandb_log = wandb_log

      self.best_valid_acc = 0

      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      # ensure deterministic behaviour
      torch.backends.cudnn.deterministic = True
      random.seed(self.random_seed)
      np.random.seed(self.random_seed)
      torch.manual_seed( self.random_seed )
      torch.cuda.manual_seed_all( self.random_seed )

   def load_tabular_data(self, inputs, targets, normalize_targets=False, split=[0.8,0.1,0.1], stratify=False):
      #
      # inputs, targets: numpy arrays
      #

      # 
      # split : [ train_split_ratio, test_split_ratio, validation_split_ratio ]
      # stratified split : [ train_split_ratio, validation_split_ratio ]
      #

      # normalize data
      scaler_inputs = MinMaxScaler()
      # scaler = StandardScaler()
      inputs_norm = scaler_inputs.fit_transform(inputs)

      if normalize_targets:
         scaler_targets = MinMaxScaler()
         targets_norm = scaler_targets.fit_transform(targets)
         targets = targets_norm

      if self.task == 'classification' and len(np.unique(targets)) > 2:
         #  for multiclass classification: labels should be LongTensor
         dataset = TensorDataset(torch.Tensor(inputs_norm), torch.LongTensor(targets)) 
      else:
         dataset = TensorDataset(torch.Tensor(inputs_norm), torch.Tensor(targets)) # regression and binary classification 

      print('dataset size:', len(dataset))
      print('dataset samples shape:', dataset.tensors[0].shape)
      print('dataset classes shape:', dataset.tensors[1].shape)

      # split dataset
      if stratify:

         temp_idx, test_idx = train_test_split(np.arange(len(targets)), train_size=split[0], random_state=self.random_seed, stratify=targets)
         train_idx, valid_idx = train_test_split(temp_idx, train_size=split[1], random_state=self.random_seed, stratify=targets[temp_idx])

         train_sampler = SubsetRandomSampler(train_idx)
         test_sampler = SubsetRandomSampler(test_idx)
         valid_sampler = SubsetRandomSampler(valid_idx)

         self.train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
         self.test_loader = DataLoader(dataset, batch_size=len(test_sampler), sampler=test_sampler)
         self.valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler)

      else:

         ds_size = dataset.tensors[0].shape[0]

         if len(split) == 3:
            test_size = int(split[1]*ds_size)
            validation_size = int(split[2]*ds_size)
            train_size = ds_size - (test_size + validation_size)
         
         if len(split) == 2: 
            ...

         if len(split) == 1: # only train_split defined: test_split = 1 - train_split; test data used for validation
            train_size = int(split[0]*ds_size)
            test_size = ds_size - train_size
            validation_size = 0

         train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

         print('\n\nSplit data:\n-----------')
         print('train data:', len(list(train_dataset)), 'samples')
         print('test data:', len(list(test_dataset)), 'samples')
         
         self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
         self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
         
         if validation_size == 0:
            print('No validation split defined, test data used for validation.') 
            self.valid_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
         else:
            print('validation data:', len(list(validation_dataset)), 'samples')
            self.valid_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

   def load_mnist(self, root_for_data, transform, train_valid_split=[50000, 10000]):
      dataset = torchvision.datasets.MNIST(root_for_data, download=True, train=True, transform=transform)
      test_dataset = torchvision.datasets.MNIST(root_for_data, download=True, train=False, transform=transform)
      print('main dataset size:', len(dataset))
      print('test dataset size:', len(test_dataset))

      train_dataset, validation_dataset = random_split(dataset, train_valid_split)
      print('train dataset size:', len(train_dataset))
      print('validation dataset size:', len(validation_dataset))

      self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
      self.valid_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)   
      self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)   

   def load_image_from_folder(self, root_for_train, root_for_test, transform, train_split=0.8):
      dataset = ImageFolder(root=root_for_train, transform=transform['train'])
      test_dataset = ImageFolder(root=root_for_test, transform=transform['test'])

      train_size = int(train_split*len(dataset))
      validation_size = len(dataset) - train_size

      train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
      
      print('main dataset size:', len(dataset))
      print('train dataset size:', len(train_dataset))
      print('validation dataset size:', len(validation_dataset))
      print('test dataset size:', len(test_dataset))

      self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
      self.valid_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)
      self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

   def calculate_accuracy(self, outputs, labels):

      if self.task == 'regression':
         # for regression
         return r2_score(labels, outputs.cpu().detach().numpy())

      if self.task == 'classification':
         if self.output_class_num == 2:
            # binary accuracy
            preds = outputs >= self.threshold
            return preds.eq(labels).sum() / labels.numel()
         else:
            # multiclass accuracy
            # _, preds = torch.max(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            return torch.tensor(torch.sum(preds == labels.view(-1)).item() / len(preds))

   def train(self, model, iterator):

      epoch_loss = 0
      epoch_acc = 0
      
      model.train()
      
      if self.progressbar:
         iterator = tqdm(iterator, ascii=True, unit='batch')
      else:
         iterator = iterator

      for (inputs, labels) in iterator:

         if self.progressbar:
            iterator.set_description('Training  ')

         if self.task == 'classification' and self.output_class_num > 2: # multiclass classification
            labels = labels.view(-1)
         
         inputs = inputs.to(self.device)
         labels = labels.to(self.device)

         if self.class_weight:
            weight = torch.ones(len(labels)).view(-1,1)
            for i,l in enumerate(labels):
               if l == 1: # if minority class
                  weight[i] = self.class_weight
         
         self.optimizer.zero_grad()
               
         preds = model(inputs)
         if self.class_weight: 
            loss = self.loss_fn(preds, labels, weight=weight)
         else:
            loss = self.loss_fn(preds, labels)
         acc = self.calculate_accuracy(preds, labels)
         loss.backward()
         self.optimizer.step()
         
         epoch_loss += loss.item()
         epoch_acc += acc.item()
         
      return epoch_loss / len(iterator), epoch_acc / len(iterator)

   def evaluate(self, model, iterator):
      
      epoch_loss = 0
      epoch_acc = 0
      
      model.eval()
      
      with torch.no_grad():

         if self.progressbar:
            iterator = tqdm(iterator, ascii=True, unit='batch')
         else:
            iterator = iterator

         for (inputs, labels) in iterator:

            if self.progressbar:
               iterator.set_description('Validation')

            if self.task == 'classification' and self.output_class_num > 2: # multiclass classification
               labels = labels.view(-1)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            preds = model(inputs)
            loss = self.loss_fn(preds, labels)
            acc = self.calculate_accuracy(preds, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
         
      return epoch_loss / len(iterator), epoch_acc / len(iterator)

   def fit(self, model, show_nth_epoch=1, save_model_path=None):

      if self.task == 'classification':
         self.output_class_num = len(np.unique(next(iter(self.train_loader))[1]))
       
      self.optimizer = self.optim(model.parameters(), self.learning_rate)

      if self.lr_sch:
         self.scheduler = self.lr_sch(self.optimizer, **self.lr_sch_args) # !!! lr_scheduler !!!
      
      history = []

      lr_current = self.learning_rate

      for epoch in range(self.epochs):
         epoch_starttime = time.time()
         train_loss, train_acc = self.train(model, self.train_loader)
         valid_loss, valid_acc = self.evaluate(model, self.valid_loader)
         epoch_time = time.time() - epoch_starttime

         if self.lr_sch:
            self.scheduler.step() # !!! lr_scheduler !!!
            lr_current = self.optimizer.state_dict()['param_groups'][0]['lr'] # !!! lr_scheduler !!!

         history.append({'train_loss':train_loss, 'train_acc':train_acc, 'valid_loss':valid_loss, 'valid_acc':valid_acc, 'lr':lr_current, 'epoch_time':epoch_time})

         if save_model_path and (valid_acc > self.best_valid_acc):
            self.best_valid_acc = valid_acc
            torch.save(model.state_dict(), save_model_path)

         if self.wandb_log:
            wandb.log({'train_loss':train_loss, 'train_acc':train_acc, 'valid_loss':valid_loss, 'valid_acc':valid_acc, 'lr':lr_current, 'epoch_time':epoch_time})

         if (epoch+1) % show_nth_epoch == 0:
            print(f'| Epoch: {epoch+1:04} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% |')
             
      return history

   def show_history(self, history):

      epochs = [i for i in range(1,len(history)+1)]
      train_loss = [data['train_loss'] for data in history]
      valid_loss = [data['valid_loss'] for data in history]
      valid_acc = [data['valid_acc'] for data in history]

      fig, ax1 = plt.subplots(figsize=(8,6), tight_layout=True)
      ax1.grid(linestyle=':')
      ax1.set_xlabel('Epochs')
      ax1.set_ylabel('Losses')
      #ax1.plot(epochs, train_loss, label='train loss', marker='.')
      ax1.plot(epochs, train_loss, label='train loss')
      #ax1.plot(epochs, valid_loss, label='val loss', marker='.')
      ax1.plot(epochs, valid_loss, label='val loss')
      ax1.legend(loc=3)
      ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
      ax2.set_ylabel('val accuracy')
      #ax2.plot(epochs, valid_acc, label='val accuracy', color='green', linestyle=':', marker='x')
      ax2.plot(epochs, valid_acc, label='val accuracy', color='green')
      ax2.legend(loc=8)
      
      if self.wandb_log:
         wandb.log({"chart": fig})
       
      plt.show()

   def show_metrics(self, model, test_loader, heatmap=False, supress_print=False):
      X_test, y_test = next(iter(test_loader))
      X_test = X_test.to(self.device)
      
      if self.task == 'classification':
         if self.output_class_num == 2: # binary classification
            if not isinstance(self.loss_fn, nn.BCEWithLogitsLoss): # eg. BCELoss
               #preds = model(X_test).cpu().round().detach().numpy() # create predictions for binary classification
               preds = (model(X_test).cpu() >= self.threshold).detach().int().numpy()
            else: # ---> BCEWithLogitsLoss
               preds = torch.sigmoid(model(X_test)).cpu().round().detach().numpy() # create predictions for binary classification 
         else:
            predictions = model(X_test).cpu().detach().numpy() # multiclass classification
            preds = torch.argmax(torch.exp(torch.Tensor(predictions)) / torch.sum( torch.exp(torch.Tensor(predictions)) ), dim=1)

         if not supress_print:
            print(classification_report(y_test, preds))
            print('\n\nAccuracy Score:', accuracy_score(y_test, preds))
            print('\n\nConfusion Matrix:\n\n', confusion_matrix(y_test, preds), '\n\n')

            if heatmap:
               ax = sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='', cmap='Blues')
               plt.ylabel('True label')
               plt.xlabel('Predicted label')
               plt.show()

            print('\n\nMatthews correlation coefficient (MCC):', matthews_corrcoef(y_test, preds))

         if self.output_class_num == 2: # binary classification
            return accuracy_score(y_test, preds), precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds)
         else:
            return accuracy_score(y_test, preds), matthews_corrcoef(y_test, preds)
      
      if self.task == 'regression':
         preds = model(X_test).cpu().detach().numpy() # regression
         print('R2 score:',r2_score(y_test, preds))
         return r2_score(y_test, preds)

   def predict(self, model, test_loader):
      X_test, y_test = next(iter(test_loader))
      
      X_test = X_test.to(self.device)
      
      # preds =  model(X_test).cpu().round().detach().numpy() # binary classification
      preds =  model(X_test).cpu().detach().numpy() # multiclass classification

      return y_test, preds

   def infer(self, model, X):
      ...