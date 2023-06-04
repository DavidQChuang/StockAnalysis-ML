import os
from tqdm import tqdm
import time

from dataclasses import dataclass
import inspect

def get_bar_format(dataset_len, batch_size):
    len_n_fmt = len(str(math.ceil((dataset_len / batch_size))))
    bar_format = '{n_fmt:>%d}/{total_fmt:%d} [{bar:30}] {elapsed} - eta: {remaining}, {rate_fmt}{postfix}' %(len_n_fmt,len_n_fmt)
    return bar_format
    
def format_loss(n):
    f = '{0:.4g}'.format(n)
    f = f.replace('+0', '+')
    f = f.replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n

def format_tensor(t, use_cuda):
    t = t.float()
    
    if use_cuda:
        t = t.cuda()
        
    return t

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

class StandardModule(ABC):
    def __init__(self, model_json, device=None, verbosity=1):
        super().__init__()
        
        self.device = device
        self.conf = StandardConfig.from_dict(model_json)
        
        if verbosity >= 1:
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
        
        train_data, valid_data = data.random_split(dataset, [train_ratio, validation_ratio])
        
        train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        
        return train_dataloader, valid_dataloader
    
    
    def get_filename(self):
        filename = self.conf.model_filename
        classname = self.__class__.__name__
        
        return f"{classname}-{filename}.ckpt"
    
    def print_validation_split(self, dataset_len):
        train_ratio = 1-self.conf.validation_split
        train_data = int(dataset_len*train_ratio)
        
        print(f"Splitting data at a {train_ratio} ratio: {train_data}/{dataset_len-train_data}")
        
    @abstractmethod
    def standard_train(self, dataset):
        pass
    
    @abstractmethod
    def save(self, filename=None):
        pass
    
    @abstractmethod
    def load(self, filename=None):
        pass
    
    def _get_training_data(self, dataset):
        return self.get_training_data(dataset)
        
class PytorchStandardModule(StandardModule, nn.Module):
    def __init__(self, model_json, device=None, verbosity=1):
        super().__init__(model_json, device, verbosity)
        
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
        self.print_validation_split(len(dataset))
        
        # Loss/optimizer functions
        loss_func, optimizer = self._setup_optimizer(model)
        
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
        bar_format = get_bar_format(len(dataset), self.conf.batch_size)
        
        start_time = time.time()
        pstart_time = time.process_time()
        for e in range(run_epochs):  # loop over the dataset multiple times
            addl_desc = '' if first_run else f'; Total epochs: {self.runtime["epoch"] + 1}/{lifetime_epochs}'
            print(f"Epoch {e+1}/{run_epochs}{addl_desc}")
            if use_cuda:
                print(f"GPU: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            
            # Scramble data
            train, valid = self._get_training_data(dataset)
            
            # Train model
            model.train(True)
            train_loss = 0.0
            train_data = []
            for data in tqdm(train, bar_format=bar_format):
                X    : torch.Tensor = format_tensor(data["X"], use_cuda)
                Y_HAT: torch.Tensor = format_tensor(data["y"], use_cuda)
                train_data.append({ "X": X, "y": Y_HAT })
                
            train_progress = tqdm(train_data, bar_format=bar_format)
            for train_iter, data in enumerate(train_progress):
                X, Y_HAT = data["X"], data["y"]

                # zero the parameter gradients
                model.zero_grad()

                # forward > backward > optimize
                y = model(X)
                loss = loss_func.forward(y, Y_HAT)
                
                loss.backward()
                optimizer.step()

                # print statistics
                train_loss += loss.item()
                if math.isnan(train_loss):
                    raise ArithmeticError("Failed training, loss = NaN")
        
                train_progress.set_postfix(loss=format_loss(train_loss / (train_iter + 1)), refresh=False)
                
            del train_data
                
            # Validate results
            model.eval()
            valid_loss = 0.0
            valid_progress = tqdm(valid, bar_format=bar_format)
            for valid_iter, data in enumerate(valid_progress):
                X    : torch.Tensor = format_tensor(data["X"], use_cuda)
                Y_HAT: torch.Tensor = format_tensor(data["y"], use_cuda)
                
                # with autocast(self.device, dtype=torch.float32):
                with torch.no_grad():
                    y = model(X)
                    loss = loss_func.forward(y, Y_HAT)

                # print statistics
                valid_loss += loss.item()
                if math.isnan(valid_loss):
                    raise ArithmeticError("Failed training, val_loss = NaN")
        
                valid_progress.set_postfix(val_loss=format_loss(valid_loss / (valid_iter + 1)), refresh=False)
                
            self.runtime['epoch'] += 1
                
            print()
            
        print(f"Done in user: {time.time() - start_time:.2f}s; sys: {time.process_time() - pstart_time:.2f}s.\n")
        self.runtime['loss'] = train_loss / train_iter
        self.runtime['val_loss'] = valid_loss / valid_iter
            
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
        
import deepspeed
        
class DeepspeedWrapper(PytorchStandardModule):
    def __init__(self, module: StandardModule, model_json, device=None, verbosity=1):
        super().__init__(model_json, device, verbosity=0)
        
        if 'deepspeed' not in model_json:
            raise Exception("'deepspeed' key must be present in model parameters.")
        
        # Copy normal model parameters
        model_json['deepspeed']['train_batch_size'] = model_json['batch_size']
        
        model_engine, optimizer, _, _ = deepspeed.initialize(config=model_json['deepspeed'],
                                                     model=module,
                                                     model_parameters=module.parameters())
        
        self.module = module
        self.model_engine = model_engine
        
    def _setup_optimizer(self, model):
        self.optimizer = self.model_engine
        
    def standard_train(self, dataset):
        print(f'> Training model Deepspeed[{self.module.__class__.__name__}].')
        self.print_validation_split(len(dataset))
            
        # Loss/optimizer functions
        loss_func: nn.MSELoss = nn.MSELoss()
        
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
        bar_format = get_bar_format(len(dataset), self.conf.batch_size)
        
        start_time = time.time()
        pstart_time = time.process_time()
        for e in range(run_epochs):  # loop over the dataset multiple times
            addl_desc = '' if first_run else f'; Total epochs: {self.runtime["epoch"] + 1}/{lifetime_epochs}'
            print(f"Epoch {e+1}/{run_epochs}{addl_desc}")
            
            # Scramble data
            train, valid = self._get_training_data(dataset)
            
            # Train model
            train_loss = 0.0
            train_data = []
            for data in tqdm(train, bar_format=bar_format):
                X    : torch.Tensor = format_tensor(data["X"], use_cuda)
                Y_HAT: torch.Tensor = format_tensor(data["y"], use_cuda)
                train_data.append({ "X": X, "y": Y_HAT })
                
            train_progress = tqdm(train_data, bar_format=bar_format)
            for train_iter, data in enumerate(train_progress):
                X, Y_HAT = data["X"], data["y"]
                
                # forward > backward > optimize
                y = self.model_engine(X)
                loss = loss_func.forward(y, Y_HAT)
                
                self.model_engine.backward(loss)
                self.model_engine.step()

                # print statistics
                train_loss += loss.item()
                if math.isnan(train_loss):
                    raise ArithmeticError("Failed training, loss = NaN")
        
                train_progress.set_postfix(loss=format_loss(train_loss / (train_iter + 1)), refresh=False)
            
            # Validate results
            valid_loss = 0.0
            valid_progress = tqdm(valid, bar_format=bar_format)
            for valid_iter, data in enumerate(valid_progress):
                X    : torch.Tensor = format_tensor(data["X"], use_cuda)
                Y_HAT: torch.Tensor = format_tensor(data["y"], use_cuda)
                
                with torch.no_grad():
                    y = self.model_engine(X)
                    loss = loss_func.forward(y, Y_HAT)

                # print statistics
                valid_loss += loss.item()
                if math.isnan(valid_loss):
                    raise ArithmeticError("Failed training, val_loss = NaN")
        
                valid_progress.set_postfix(val_loss=format_loss(valid_loss / (valid_iter + 1)), refresh=False)
                
            self.runtime['epoch'] += 1
                
            print()
            
        print(f"Done in user: {time.time() - start_time:.2f}s; sys: {time.process_time() - pstart_time:.2f}s.\n")
        self.runtime['loss'] = train_loss / train_iter
        self.runtime['val_loss'] = valid_loss / valid_iter
            
    def get_filename(self):
        return "ds-" + self.module.get_filename()
    
    def save(self, filename=None):
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        dir = os.path.dirname(filename)
        file = os.path.basename(filename)
        
        self.model_engine.save_checkpoint(dir, file, client_state=self.runtime)
            
    def load(self, filename=None):
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        dir = os.path.dirname(filename)
        file = os.path.basename(filename)
        
        _, self.runtime = self.model_engine.load_checkpoint(dir, file)