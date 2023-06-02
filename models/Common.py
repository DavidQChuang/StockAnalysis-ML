from tqdm import tqdm

from dataclasses import dataclass
import inspect

@dataclass(frozen=True)
class StandardConfig:
    # https://stackoverflow.com/questions/54678337/how-does-one-ignore-extra-arguments-passed-to-a-dataclass
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    # Trainer params
    loss        : str = "mean_squared_error"
    optimizer   : str = "adam"

    test_split        : float = 0.1
    validation_split  : float = 0.2
    batch_size          : int = 64
    epochs              : int = 15
    
    # General NN architecture params
    hidden_layer_size   : int = 50
    dropout_rate        : int = 0.3
    
    # Model I/O parameters
    seq_len             : int = 24
    out_seq_len         : int = 1
    
    @property
    def model_filename(self):
        return "%d_%d+%d" % (
            self.hidden_layer_size, self.seq_len, self.out_seq_len)
        

import torch
import math
from torch import nn, optim, autocast
from torch.utils import data
from abc import ABC, abstractmethod

class StandardModule(nn.Module, ABC):
    def __init__(self, model_json, device=None):
        super().__init__()
        
        self.device = device
        self.conf = StandardConfig.from_dict(model_json)
        print("> Model config: ")
        print(self.conf)
        print()
        
    # def get_training_data(self, dataset):
    #     validation_ratio = self.conf.validation_split
    #     train_ratio = 1 - validation_ratio
        
    #     batch_size = self.conf.batch_size
        
    #     print("> Data: ")
    #     print(f"Window: {self.conf.seq_len}+{self.conf.out_seq_len}")
        
    #     print(f"Splitting data at a {train_ratio} ratio: ", end="")
    #     train_data, valid_data = data.random_split(dataset, [train_ratio, validation_ratio])
        
    #     print(f"{len(train_data)}/{len(valid_data)}")
    #     train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    #     valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        
    #     print()
        
    #     return train_dataloader, valid_dataloader
    def get_training_data(self, dataset):
        validation_ratio = self.conf.validation_split
        train_ratio = 1 - validation_ratio
        
        batch_size = self.conf.batch_size
        
        print(f"Splitting data at a {train_ratio} ratio: ", end="")
        train_data, valid_data = data.random_split(dataset, [train_ratio, validation_ratio])
        
        print(f"{len(train_data)}/{len(valid_data)}")
        train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        
        return train_dataloader, valid_dataloader
    
    
    def get_filename(self):
        filename = self.conf.model_filename
        classname = self.__class__.__name__
        
        return f"{classname}-{filename}.ckpt"
        
    @abstractmethod
    def standard_train(self, dataset):
        pass
    
    @abstractmethod
    def save(self, filename=None):
        pass
    
    @abstractmethod
    def load(self, filename=None):
        pass
    
    @abstractmethod
    def _setup_optimizer(self):
        pass
    
    def _get_training_data(self, dataset):
        return self.get_training_data(dataset)
        
class PytorchStandardModule(StandardModule):
    def __init__(self, model_json, device=None):
        super().__init__(model_json, device)
        
        self.loss_func: nn.MSELoss = nn.MSELoss()
        self.optimizer: optim.Optimizer = None
        
        self.ckpt = None
        self.runtime = None
        
    def _setup_optimizer(self, model):
        if not self.optimizer:
            self.optimizer = optim.Adam(model.parameters())
            
        return self.loss_func, self.optimizer
            
    def standard_train(self, dataset):
        model = self

        print(f'> Training model {self.__class__.__name__}.')
        
        loss_func, optimizer = self._setup_optimizer(model)
        
        len_n_fmt = len(str(math.ceil((len(dataset) / self.conf.batch_size))))
        bar_format = '{n_fmt:>%d}/{total_fmt:%d} [{bar:30}] {elapsed} - eta: {remaining}, {rate_fmt}{postfix}' %(len_n_fmt,len_n_fmt)
        def format_loss(n):
            f = '{0:.4g}'.format(n)
            f = f.replace('+0', '+')
            f = f.replace('-0', '-')
            n = str(n)
            return f if len(f) < len(n) else n
        
        first_run = self.runtime == None
        if first_run:
            self.runtime = {
                'epoch': 0,
                'loss': 0,
                'val_loss': 0
            }
            
        use_cuda = self.device == "cuda"
        
        run_epochs = self.conf.epochs
        lifetime_epochs = self.runtime["epoch"] + run_epochs
        for e in range(run_epochs):  # loop over the dataset multiple times
            ep = e+1
            addl_desc = '' if first_run else f'; Total epochs: {self.runtime["epoch"] + 1}/{lifetime_epochs}'
            desc_train = f"Epoch {ep}/{run_epochs}{addl_desc}"
            desc_valid = f"Validating"
            
            print(desc_train)
            
            # Scramble data
            train, valid = self._get_training_data(dataset)
            
            # Train model
            model.train(True)
            train_loss = 0.0
            train_progress = tqdm(train, bar_format=bar_format)
            for i, data in enumerate(train_progress):
                X    : torch.Tensor = data["X"]
                Y_HAT: torch.Tensor = data["y"]
                
                X = X.float()
                Y_HAT= Y_HAT.float()
                
                if use_cuda:
                    X, Y_HAT = X.cuda(), Y_HAT.cuda()

                # zero the parameter gradients
                model.zero_grad()
                # optimizer.zero_grad()
                # criterion.zero_grad()

                # forward > backward > optimize
                # with autocast(self.device, dtype=torch.float32):
                y = model(X)
                loss = loss_func.forward(y, Y_HAT)
                
                loss.backward()
                optimizer.step()

                # print statistics
                train_loss += loss.item()
        
                train_progress.set_postfix(loss=format_loss(train_loss / (i + 1)), refresh=False)
                
            self.runtime['loss'] = train_loss / i
            
            # Validate results
            model.eval()
            valid_loss = 0.0
            valid_progress = tqdm(valid, bar_format=bar_format)
            for i, data in enumerate(valid_progress):
                X    : torch.Tensor = data["X"]
                Y_HAT: torch.Tensor = data["y"]
                
                X = X.float()
                Y_HAT= Y_HAT.float()
                
                if use_cuda:
                    X, Y_HAT = X.cuda(), Y_HAT.cuda()
                
                # with autocast(self.device, dtype=torch.float32):
                with torch.no_grad():
                    y = model(X)
                    loss = loss_func.forward(y, Y_HAT)

                # print statistics
                valid_loss += loss.item()
                # valid_loss = loss.item() * X.size(0)
        
                valid_progress.set_postfix(val_loss=format_loss(valid_loss / (i + 1)), refresh=False)
                
            self.runtime['val_loss'] = valid_loss / i
            self.runtime['epoch'] += 1
                
            print()
            
    def save(self, filename=None):
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        
        print("> Saving model: ")
        
        if self.runtime == None:
            raise Exception("Cannot save model without running it first.")
        
        ckpt = {
            'epoch': self.runtime["epoch"],
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': self.runtime["loss"],
            'val_loss': self.runtime["val_loss"]
        }
        print({
            'epoch': ckpt['epoch'],
            'loss': ckpt['loss'],
            'val_loss': ckpt['val_loss']
            })
        print()
        
        torch.save(ckpt, filename)
            
    def load(self, filename=None):
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        
        print("> Loading model: ")
        
        ckpt = torch.load(filename)
        
        self._setup_optimizer(self)
        
        self.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        
        self.runtime = {
            'epoch': ckpt['epoch'],
            'loss': ckpt['loss'],
            'val_loss': ckpt['val_loss'],
        }
        
        print(self.runtime)
        print()