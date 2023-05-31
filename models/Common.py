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
    def model_filename(self, id):
        return "%s%d_%d+%d" % (
            id, self.hidden_layer_size, self.in_window, self.out_window)
        

import torch
from torch import nn, optim, cuda
from torch.utils import data

class StandardModule(nn.Module):
    def __init__(self, model_json):
        super().__init__()
        
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
    
    def _setup_optimizer(self):
        pass
    
    def _get_training_data(self, dataset):
        return self.get_training_data(dataset)
        
    def train(self, dataset):
        pass
        
class PytorchStandardModule(StandardModule):
    def __init__(self, model_json):
        super().__init__(model_json)
        
        self.loss_func: nn.MSELoss = nn.MSELoss()
        self.optimizer: optim.Optimizer = None
        
    def _setup_optimizer(self, model):
        if not self.optimizer:
            self.optimizer = optim.Adam(model.parameters())
            
        return self.loss_func, self.optimizer
            
    def standard_train(self, dataset):
        epochs = self.conf.epochs
        
        model = self
        if cuda.is_available():
            model.cuda()

        print('> Training model.')
        
        loss_func, optimizer = self._setup_optimizer(model)
        train, valid = self._get_training_data(dataset)
        
        len_n_fmt = len(str(len(train)))
        bar_format = '{n_fmt:>%d}/{total_fmt:%d} [{bar:30}] [{elapsed} - eta: {remaining}, {rate_fmt}{postfix}]' %(len_n_fmt,len_n_fmt)
        def format_loss(n):
            f = '{0:.5g}'.format(n)
            f = f.replace('+0', '+')
            f = f.replace('-0', '-')
            n = str(n)
            return f if len(f) < len(n) else n
        
        for e in range(epochs):  # loop over the dataset multiple times
            desc_train = f"Epoch {e+1}/{epochs}"
            desc_valid = f"Validating"
            
            print(desc_train)
            
            model.train(True)
            train_loss = 0.0
            train_progress = tqdm(train, total=len(train), bar_format=bar_format)
            for i, data in enumerate(train_progress):
                X    : torch.Tensor = data["X"]
                Y_HAT: torch.Tensor = data["y"]
                
                X = X.float()
                Y_HAT= Y_HAT.float()
                
                if cuda.is_available():
                    X, Y_HAT = X.cuda(), Y_HAT.cuda()

                # zero the parameter gradients
                model.zero_grad()
                # optimizer.zero_grad()
                # criterion.zero_grad()

                # forward > backward > optimize
                y = model(X)
                loss = loss_func.forward(y, Y_HAT)
                
                loss.backward()
                optimizer.step()

                # print statistics
                train_loss += loss.item()
        
                if i % 5 == 4:
                    train_progress.set_postfix(loss=format_loss(train_loss / (i + 1)), refresh=False)
                    train_progress.update(5)
            
            model.eval()
            valid_loss = 0.0
            valid_progress = tqdm(valid, total=len(valid), bar_format=bar_format)
            for i, data in enumerate(valid_progress):
                X = data["X"].float()
                Y_HAT = data["y"].float()
                if cuda.is_available():
                    X, Y_HAT = X.cuda(), Y_HAT.cuda()
                
                y = model(X)
                loss = loss_func.forward(y, Y_HAT)

                # print statistics
                valid_loss += loss.item()
                # valid_loss = loss.item() * X.size(0)
        
                valid_progress.set_postfix(val_loss=format_loss(train_loss / (i + 1)), refresh=False)
                valid_progress.update()
                
            print()
            train, valid = self._get_training_data(dataset)