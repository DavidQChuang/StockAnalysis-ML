import os
import time
from tqdm import tqdm

from dataclasses import dataclass
import inspect

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets.Common import AdvancedTimeSeriesDataset, TimeSeriesDataset
        
import torch
import math
from torch import nn, optim, autocast
from torch.utils import data
from abc import ABC, abstractmethod

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

def count_parameters(model):
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params+=params
        
    # print(table)
    print(f"Total trainable params: {total_params}")
    return total_params

@dataclass(frozen=True)
class ModelConfig:
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
    hidden_layer_size   : int   = 50
    dropout_rate        : float = 0.3
    
    # Model I/O parameters
    seq_len             : int = 24
    out_seq_len         : int = 1
    
    precompile          : bool = False
    pin_memory          : bool = False
    
    indicators          : list[dict] = None
    columns             : list[dict] = None
    
    @property
    def column_names(self):
        return [ col['name'] for col in self.columns ]
    
    @property
    def model_filename(self):
        return "%d_%d+%df%d" % (
            self.hidden_layer_size, self.seq_len, self.out_seq_len, len(self.columns))

class StandardModel(ABC):
    def __init__(self, model_json, device=None, verbosity=1):
        super().__init__()
        
        self.device = device
        self.conf = ModelConfig.from_dict(model_json)
        self.scaler = StandardScaler()
        
        if verbosity >= 1:
            print("> Model config: ")
            print(self.conf)
            print()
    
    def print_validation_split(self, dataset_len):
        train_ratio = 1-self.conf.validation_split
        train_data = int(dataset_len*train_ratio)
        
        print(f"Splitting data at a {train_ratio} ratio: {train_data}/{dataset_len-train_data}")
        
    def scale_dataset(self, dataset: TimeSeriesDataset, fit=False):
        print('Fitting dataset.')
        print('Before: ', dataset.df[dataset.column_names][:3].to_numpy(), 'dtype=', dataset.df['close'].dtype)
        
        if isinstance(dataset, AdvancedTimeSeriesDataset):
            columns_to_scale = [ col['name'] for col in dataset.columns if 'is_scaled' in col and col['is_scaled']]
        else:
            columns_to_scale = dataset.column_names
        
        if fit:
            dataset.df[columns_to_scale] = self.scaler.fit_transform(dataset.df[columns_to_scale])
        else: 
            dataset.df[columns_to_scale] = self.scaler.transform(dataset.df[columns_to_scale])
        
        print('After: ', dataset.df[dataset.column_names][:3].to_numpy(), 'dtype=', dataset.df['close'].dtype)
        print()
        
        return dataset
    
    def scale_input(self, input):
        """
        Scales an unscaled input x into the standard distribution this model was fitted to.
            z = (x - u) / s
        """
        return (input - self.scaler.mean_[0]) / self.scaler.scale_[0]
    
    def scale_output(self, output):
        """
        Unscales a scaled output z into unscaled units.
            x = z * s + u
        """
        return output * self.scaler.scale_[0] + self.scaler.mean_[0]
        
    @abstractmethod
    def standard_train(self, dataset):
        pass
    
    @abstractmethod
    def infer(self, X, scale_inputs=False, scale_outputs=False):
        pass
    
    @abstractmethod
    def save(self, filename=None):
        pass
    
    @abstractmethod
    def load(self, filename=None):
        pass
    
    def get_model_name(self):
        return self.__class__.__name__
    
    def get_filename(self):
        filename = self.conf.model_filename
        classname = self.get_model_name()
        
        return f"{classname}-{filename}.ckpt"
    
    @property
    def pin_memory(self):
        return (self.device != None and not self.device.startswith('cpu') and self.conf.pin_memory == True)
        
    def get_training_data(self, dataset: TimeSeriesDataset):
        validation_ratio = self.conf.validation_split
        train_ratio = 1 - validation_ratio
        
        batch_size = self.conf.batch_size
        
        train_data, valid_data = data.random_split(dataset, [train_ratio, validation_ratio])
        
        #https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
        # If pinning, tensors on CPU remain in non-paged memory.
        # This can speed up calls to Tensor.cuda()
        #   and allows async calls with Tensor.cuda(non_blocking=True).
        if self.pin_memory:
            print("Pinning")
            train_dataloader = data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=False, pin_memory=True)
            valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size,
                                               shuffle=False, pin_memory=True)
        # If not pinning, use the dataset collate_fn that converts data
        #   in numpy form to tensors on the correct device.
        else:
            print("Unpinned")
            train_dataloader = data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=dataset.get_collate_fn(device=self.device),
                                               shuffle=False)
            valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size,
                                               collate_fn=dataset.get_collate_fn(device=self.device),
                                               shuffle=False)
        
        return train_dataloader, valid_dataloader
        
class PytorchModel(StandardModel):
    def __init__(self, network: nn.Module, model_json: dict, device=None, verbosity=1):
        super().__init__(model_json, device, verbosity)
        
        self.ckpt = None
        self.runtime = None
        self.module = network
        self.model_name = network.__class__.__name__
        
        torch.set_float32_matmul_precision('high')
        
        self.module = self.module.float()
        if device != None and not device.startswith('cpu'):
            self.module = self.module.to(device)
            
        if self.conf.precompile == True:
            self.module = torch.compile(self.module, mode="reduce-overhead")
            print("> Compiling model.")
        
        self.optimizer_state = None
        
    def get_model_name(self):
        return self.model_name
    
    def get_loss_func(self):
        return nn.SmoothL1Loss()
    
    def get_optimizer(self):
        optimizer = optim.Adam(self.module.parameters(), lr=0.0005, weight_decay=0.0001)
        
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        
        return optimizer
            
    def single_train(self, X, Y_HAT, loss_func, optimizer):
        """Performs a single forward and backward step with the optimizer and a loss calculation.
        Model should be in training mode before this is run.

        Args:
            X (torch.Tensor): Input vector, must be same device as module.
            Y_HAT (torch.Tensor): Expected output vector, must be same device as module.
            loss_func (torch.nn._Loss): Loss function.
            optimizer (torch.optim.Optimizer): The optimizer, must be initialized with module.parameters.

        Returns:
            (torch.Tensor, torch.Tensor): The output value and the loss.
        """
        # zero the parameter gradients
        self.module.zero_grad()

        # forward > backward > optimize
        y    : torch.Tensor = self.module(X)
        loss : torch.Tensor = loss_func.forward(y, Y_HAT)
        
        if not math.isnan(loss.item()):
            loss.backward()
            optimizer.step()
        
        return y, loss
            
    def single_infer(self, X, Y_HAT, loss_func):
        """Performs a single inference and loss calculation with gradients off.
        Model should be in evaluation mode before this is run.

        Args:
            X (torch.Tensor): Input vector, must be same device as module.
            Y_HAT (torch.Tensor): Expected output vector, must be same device as module.
            loss_func (torch.nn._Loss): Loss function.

        Returns:
            (torch.Tensor, torch.Tensor): The output value and the loss.
        """
        with torch.no_grad():
            y    : torch.Tensor = self.module(X)
            loss : torch.Tensor = loss_func.forward(y, Y_HAT)
            
            return y, loss
    
    def infer(self, X, scale_inputs=False, scale_outputs=False):
        with torch.no_grad():
            if scale_inputs:
                input = self.scale_input(X)
            else:
                input = X
                    
            output = self.module.forward(input)
            
            if scale_outputs:
                output = self.scale_output(output)
                
            return output
    
    def standard_train(self, dataset: TimeSeriesDataset):
        if not isinstance(dataset, TimeSeriesDataset):
            raise TypeError("Dataset must be TimeSeriesDataset.")
        
        if self.conf.epochs == 0:
            print(f'> Zero epochs. Skipping training model {self.get_model_name()}.')
            print()
            return
        
        # -- Setup
        use_cuda = self.device != None and self.device != "cpu"
        module: nn.Module = self.module

        print(f'> Training model {self.get_model_name()}.')
        count_parameters(module)
        print()
        self.print_validation_split(len(dataset))
        
        # Scale data
        print()
        dataset = self.scale_dataset(dataset, True) # dataset still numpy
        real_loss_scale = self.scaler.scale_[0]
        
        # Loss/optimizer functions
        loss_func = self.get_loss_func()
        optimizer = self.get_optimizer()
        
        # Runtime logging stuff
        first_run = self.runtime == None
        if first_run:
            self.runtime = {
                'epoch': 0,
                'loss': 0,
                'val_loss': 0
            }
        
        # Current session epochs & lifetime epochs
        run_epochs = self.conf.epochs
        lifetime_epochs = self.runtime["epoch"] + run_epochs
        
        # Time realtime and cpu time
        start_time = time.time()
        pstart_time = time.process_time()
        bar_format = get_bar_format(len(dataset), self.conf.batch_size)
        
        # -- Training
        for e in range(run_epochs):
            # Logging stuff
            addl_desc = '' if first_run else f'; Total epochs: {self.runtime["epoch"] + 1}/{lifetime_epochs}'
            print(f"Epoch {e+1}/{run_epochs}{addl_desc}")
            if use_cuda:
                print(f"GPU: {torch.cuda.memory_allocated() / 1024**2:.2f}MB ", end='')
                
            # Scramble data, this converts the numpy dataset into Tensors
            train, valid = self.get_training_data(dataset)
            
            # Pin memory
            train_data = []
            if self.pin_memory == True:
                for data in tqdm(train, desc="Pinning", bar_format=bar_format):
                    X    : torch.Tensor = data["X"].float().to(self.device)
                    Y_HAT: torch.Tensor = data["y"].float().to(self.device)
                    train_data.append({ "X": X, "y": Y_HAT })
            else:
                # Putting the data in a list seems to be slightly faster than
                # iterating over the dataset directly for some reason
                for data in tqdm(train, desc="Pinning", bar_format=bar_format):
                    X    : torch.Tensor = data["X"]
                    Y_HAT: torch.Tensor = data["y"]
                    train_data.append({ "X": X, "y": Y_HAT })
            
            # Train model
            module.train(True)
            train_loss = 0.0
                    
            train_progress = tqdm(train_data, bar_format=bar_format)
            for train_iter, data in enumerate(train_progress):
                X, Y_HAT = data["X"], data["y"]

                # zero the parameter gradients
                y, loss = self.single_train(X, Y_HAT, loss_func, optimizer)

                # print statistics
                train_loss += loss.item()
                if math.isnan(train_loss):
                    print(">>> IN/OUT: ")
                    print(X, Y_HAT)
                    print()
                    print()
                    try:
                        with torch.autograd.detect_anomaly():
                            y = self.module.forward(X)
                            torch.mean(y).backward()
                    except Exception as e:
                        print(e)
                        
                    print()
                    print()
                    print(">>> PARAMETERS: ")
                    weights = optimizer.param_groups[0]['params']
                    weights_flat = [torch.flatten(weight) for weight in weights]
                    weights_1d = torch.cat(weights_flat)
                    assert not torch.isnan(weights_1d).any()
                    assert not torch.isinf(weights_1d).any()
                    print(f"max params: {weights_1d.max()}, min: {weights_1d.min()}")
                    
                    grad_flat = [torch.flatten(weight.grad) for weight in weights if weight.grad != None]
                    if grad_flat != []:
                        grad_1d = torch.cat(grad_flat)
                        assert not torch.isnan(grad_1d).any()
                        assert not torch.isinf(grad_1d).any()
                        print(f"max grad: {grad_1d.max()}, min: {grad_1d.min()}")
                    
                    for p in list(filter(lambda p: p.grad is not None, self.module.parameters())):
                        print(p.grad.data.norm(2).item())
                    
                    # print("\nSaving failed model separately for debugging: ")
                    # self.save("ckpt/fail_" + self.get_filename())
                    raise ArithmeticError("Failed training, loss = NaN")
        
                # loss = (scale(x) - scale(y))^2
                # x = inv_scale(sqrt(loss))
                loss=train_loss / (train_iter + 1)
                err=math.sqrt(loss)*real_loss_scale
                train_progress.set_postfix({
                    "loss": format_loss(loss),
                    "loss($)": format_loss(err)}, refresh=False)
                
            # Validate results
            module.eval()
            valid_loss = 0.0
            valid_progress = tqdm(valid, bar_format=bar_format)
            with torch.no_grad():
                for valid_iter, data in enumerate(valid_progress):
                    if self.pin_memory:
                        X    : torch.Tensor = data["X"].float().to(self.device)
                        Y_HAT: torch.Tensor = data["y"].float().to(self.device)
                    else:
                        X    : torch.Tensor = data["X"]
                        Y_HAT: torch.Tensor = data["y"]
                    
                    y, loss = self.single_infer(X, Y_HAT, loss_func)

                    # print statistics
                    valid_loss += loss.item()
                    if math.isnan(valid_loss):
                        print("\nSaving failed model separately for debugging: ")
                        self.save("ckpt/fail_" + self.get_filename())
                        raise ArithmeticError("Failed training, val_loss = NaN")
            
                    val_loss = valid_loss / (valid_iter + 1)
                    val_err=math.sqrt(val_loss)*real_loss_scale
                    valid_progress.set_postfix({
                        "val_loss": format_loss(val_loss), "val_loss($)": format_loss(val_err) }, refresh=False)
                
            self.runtime['epoch'] += 1
                
            print()
            
        print(f"Done in user: {time.time() - start_time:.2f}s; sys: {time.process_time() - pstart_time:.2f}s.\n")
        self.runtime['loss'] = train_loss / train_iter
        self.runtime['val_loss'] = valid_loss / valid_iter
        self.optimizer_state = None if optimizer is None else optimizer.state_dict()
            
    def save(self, filename=None):
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        
        print("> Saving model: ")
        
        if self.runtime == None:
            raise Exception("Cannot save model without running it first.")
        
        ckpt = {
            'epoch': self.runtime["epoch"],
            'model_state': self.module.state_dict(),
            'optimizer_state': self.optimizer_state,
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
        
        self.module.load_state_dict(ckpt['model_state'])
        self.optimizer_state = ckpt['optimizer_state']
        
        self.runtime = {
            'epoch': ckpt['epoch'],
            'loss': ckpt['loss'],
            'val_loss': ckpt['val_loss'],
        }
        
        print(self.runtime)
        print()
        
import deepspeed
        
class DeepspeedModel(PytorchModel):
    def __init__(self, module: StandardModel, model_json, device=None, verbosity=1):
        super().__init__(module, model_json, device, verbosity=0)
        
        if 'deepspeed' not in model_json:
            raise Exception("'deepspeed' key must be present in model parameters.")
        
        # Copy normal model parameters
        model_json['deepspeed']['train_batch_size'] = model_json['batch_size']
        
        model_engine, optimizer, _, _ = deepspeed.initialize(config=model_json['deepspeed'],
                                                     model=self.module,
                                                     model_parameters=self.module.parameters())
        
        self.model_engine = model_engine
        self.optimizer = optimizer
        
    def get_optimizer(self, model):
        return None
        
    def single_train(self, X, Y_HAT, loss_func, optimizer):
        y = self.model_engine(X)
        loss = loss_func.forward(y, Y_HAT)
        
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        return y, loss
        
    def single_infer(self, X, Y_HAT, loss_func):
        y = self.model_engine(X)
        loss = loss_func.forward(y, Y_HAT)
        
        return y, loss
            
    def get_filename(self):
        return "ds-" + super().get_filename()
    
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