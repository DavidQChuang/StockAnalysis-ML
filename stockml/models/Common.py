from collections import defaultdict
import os
import time
from tqdm import tqdm

from dataclasses import dataclass, field
import inspect

from sklearn.preprocessing import StandardScaler

from stockml.datasets.Common import AdvancedTimeSeriesDataset, TimeSeriesDataset, get_scaled_column_names, get_input_column_names
import models.loss as loss
        
import torch
import math
from torch import nn, optim
from torch.amp.autocast_mode import autocast
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

    epochs              : int = 15
    
    learning_rate       : float = 0.001
    weight_decay        : float = 0.0001
    
    # General NN architecture params
    hidden_layer_size   : int   = 50
    dropout_rate        : float = 0.3
    
    # Model I/O parameters
    seq_len             : int = 24
    out_seq_len         : int = 1
    
    precompile          : bool = False
    pin_memory          : bool = False
    
    indicators          : list[dict] = field(default_factory=lambda: []) 
    columns             : list[dict] = field(default_factory=lambda: [ { "name": "close" } ])
        
    # TODO: convert shared data (between this and DatasetConfig) to an actual shared class
    # to avoid code duplication
    @property
    def column_names(self):
        return [ col['name'] for col in self.columns ]
    
    @property
    def scaled_column_names(self):
        '''
        Returns all column names where is_scaled is true or not present (i.e. default if not present is true)
        '''
        return get_scaled_column_names(self.columns)
    
    @property
    def input_column_names(self):
        '''
        Returns all column names where is_input is present and true (i.e. default if not present is false)
        '''
        return get_input_column_names(self.columns)
    
    @property
    def model_filename(self):
        return "%d_%d+%df%d" % (
            self.hidden_layer_size, self.seq_len, self.out_seq_len, len(self.columns))

class StandardModel(ABC):
    def __init__(self, model_json, device=None, verbosity=1):
        super().__init__()
        
        self.device:str = device or 'cpu'
        self.conf = ModelConfig.from_dict(model_json)
        self.scaler = StandardScaler(copy=True)
        
        if verbosity >= 2:
            print("> Model config: ")
            print(self.conf)
            print()
    
    @abstractmethod
    def standard_train(self, dataset, iter_callback=None, data_callback=None, epoch_callback=None):
        pass
    
    @abstractmethod
    def infer(self, X, scale_inputs=True, scale_outputs=True) -> torch.Tensor:
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
    
    @property
    @abstractmethod 
    def scaled_column_names(self):
        pass
        
    def scale_input(self, input, column:str|int=0, delta=False):
        """
        Scales an unscaled input column x into the normalized distribution the given column was fitted to.\n
        If the input is the difference between two unscaled inputs, set delta to True.
            z = (x - u) / s
        """
            
        if self.scaler is None or not hasattr(self.scaler, 'mean_') or self.scaled_column_names is None:
            raise RuntimeError("Scaler must be fitted before being used to scale/unscale input. "+
                               "self.scaler or scaled_column_names is None, or the scaler has not been fitted."+
                               "Run TimeSeriesDataset.scale_dataset(scaler, columns_to_scale, fit=True) first.")
            
        # if column is str, convert to index
        if type(column) is str:
            index:int = self.scaled_column_names.index(column)
        else:
            index:int = column # type: ignore
        
        if delta == True:
            return input / self.scaler.scale_[index] # type: ignore ; if the scaler has mean_ it should have everything else too
        else:
            return (input - self.scaler.mean_[index]) / self.scaler.scale_[index] # type: ignore
    
    def scale_output(self, output, column:str|int=0, is_delta=False):
        """
        Unscales a scaled output z corresponding to the given column into unscaled units.\n
        If the input is the difference between two scaled outputs, set delta to True.
            x = z * s + u
        """
            
        if self.scaler is None or not hasattr(self.scaler, 'mean_') or self.scaled_column_names is None:
            raise RuntimeError("Scaler must be fitted before being used to scale/unscale input. "+
                               "self.scaler or scaled_column_names is None, or the scaler has not been fitted."+
                               "Run TimeSeriesDataset.scale_dataset(scaler, columns_to_scale, fit=True) first.")
        
        # if column is str, convert to index
        if type(column) is str:
            index:int = self.scaled_column_names.index(column)
        else:
            index:int = column # type: ignore
        
        if is_delta == True:
            return output * self.scaler.scale_[index] # type: ignore ; if the scaler has mean_ it should have everything else too
        else:
            return output * self.scaler.scale_[index] + self.scaler.mean_[index] # type: ignore

class PytorchModel(StandardModel):
    runtime: dict | None
    
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
        
    @property
    def scaled_column_names(self):
        return self.conf.scaled_column_names
        
    def get_model_name(self):
        return self.model_name
    
    def get_loss_func(self):
        loss_funcs = self.conf.loss.split('+')
        loss_instances = []
        
        for func in loss_funcs:
            func = func.strip()
            
            if func == 'mean_squared_error' or func == 'mse':
                loss_instances.append(nn.MSELoss())
            elif func == 'mean_absolute_directional' or func == 'madl':
                loss_instances.append(loss.MADLoss())
            elif func == 'smooth_l1_loss' or func == 'smooth_l1':
                loss_instances.append(nn.SmoothL1Loss())
            else:
                raise Exception("Invalid loss function " + func)
                
        if len(loss_funcs) == 1:
            return loss_instances[0]
        else:
            return loss.CombinedLoss(loss_instances)
        # return nn.SmoothL1Loss()
    
    def get_optimizer(self):
        optimizer = optim.Adam(self.module.parameters(), lr=self.conf.learning_rate, weight_decay=self.conf.weight_decay)
        
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        
        return optimizer
            
    def single_train(self, X, Y, loss_func, optimizer):
        """Performs a single forward and backward step with the optimizer and a loss calculation.
        Model should be in training mode before this is run.

        Args:
            X (torch.Tensor): Input vector, must be same device as module.
            Y (torch.Tensor): Expected output vector, must be same device as module.
            loss_func (torch.nn._Loss): Loss function.
            optimizer (torch.optim.Optimizer): The optimizer, must be initialized with module.parameters.

        Returns:
            (torch.Tensor, torch.Tensor): The output value and the loss.
        """
        # zero the parameter gradients
        self.module.zero_grad()

        # forward > backward > optimize
        y_hat : torch.Tensor = self.module(X)
        loss  : torch.Tensor = loss_func.forward(y_hat, Y)
        
        if not math.isnan(loss.item()):
            loss.backward()
            optimizer.step()
        
        return y_hat, loss
            
    def single_infer(self, X, Y, loss_func):
        """Performs a single inference and loss calculation with gradients off.
        Model should be in evaluation mode before this is run.

        Args:
            X (torch.Tensor): Input vector, must be same device as module.t
            Y_HAT (torch.Tensor): Expected output vector, must be same device as module.
            loss_func (torch.nn._Loss): Loss function.

        Returns:
            (torch.Tensor, torch.Tensor): The output value and the loss.
        """
        with torch.no_grad():
            y_hat : torch.Tensor = self.module(X)
            loss  : torch.Tensor = loss_func.forward(y_hat, Y)
            
            return y_hat, loss
    
    def infer(self, X, scale_inputs=True, scale_outputs=True):
        with torch.no_grad():
            if scale_inputs:
                input = self.scale_input(X)
            else:
                input = X
                    
            output = self.module.forward(torch.Tensor(input).to(self.device))
            
            if scale_outputs:
                output = self.scale_output(output)
                
            return output
        
    def transform_input(self, x, inplace=True):
        # b: batch_size, n: seq_len, f: features, d: embedding_size, o: output_size, h: d_ffn
        # Make sure x is shape (batch_size, seq_len, features)
        # This unsqueezes x from (n) to (1, n, 1)
        if len(x.shape) == 1:
            x = x[None, :, None]
        if len(x.shape) == 2:
            # This unsqueezes x from (n, f) to (1, n, f)
            x = x.unsqueeze(0)
            
        if not inplace:
            x = x.clone()
        
        # Assume 'close' is the 1st column
        input_offset = x[:, 0, 0].unsqueeze(-1).clone().detach()
        # output_offset = x[:, -1, 0].unsqueeze(-1).clone().detach()
        
        # -- Offset
        # INPUT: x:                     (b, n, f)
        # INPUT: input_offset:          (b, 1)
        # Leave the 1st close value alone, since this will just set this to zero
        x[:, :, 0] = x[:, :, 0] - input_offset
        x[:, 0, 0] = input_offset
        
        return x
    
    def standard_train(self, dataset: TimeSeriesDataset, iter_callback=None, data_callback=None, epoch_callback=None):
        if not isinstance(dataset, TimeSeriesDataset):
            raise TypeError("Dataset must be TimeSeriesDataset.")
        
        if self.conf.epochs == 0:
            print(f'> Zero epochs. Skipping training model {self.get_model_name()}.')
            print()
            return
        
        # -- Setup
        use_cuda = self.device != None and self.device != "cpu"
        module: nn.Module = self.module # type: ignore

        print(f'> Training model {self.get_model_name()}.')
        count_parameters(module)
        print()
        
        # Runtime logging stuff
        first_run = self.runtime == None
        if self.runtime == None:
            self.runtime = {
                'epoch': 0,
                'loss': 0,
                'val_loss': 0
            }
        
        # Scale data (fit only if scaler is not already fit)
        should_fit_data = not hasattr(self.scaler, "mean_")
        dataset = dataset.scale_dataset(self.scaler, should_fit_data) # dataset still numpy
        if should_fit_data:
            self.runtime['columns'] = self.conf.column_names
            self.runtime['mean_'] = self.scaler.mean_
            self.runtime['var_'] = self.scaler.var_
            self.runtime['scale_'] = self.scaler.scale_
            
        print("> Target column for regression:", dataset.target)
        
        # Loss/optimizer functions
        loss_func = self.get_loss_func()
        optimizer = self.get_optimizer()
        
        # Current session epochs & lifetime epochs
        run_epochs = self.conf.epochs
        lifetime_epochs = self.runtime["epoch"] + run_epochs
        
        # Time realtime and cpu time
        start_time = time.time()
        pstart_time = time.process_time()
        bar_format = get_bar_format(len(dataset), dataset.batch_size)
        
        # -- Training
        for e in range(run_epochs):
            # Logging stuff
            addl_desc = '' if first_run else f'; Total epochs: {self.runtime["epoch"] + 1}/{lifetime_epochs}'
            print(f"Epoch {e+1}/{run_epochs}{addl_desc}")
            if use_cuda:
                print(f"GPU: {torch.cuda.memory_allocated() / 1024**2:.2f}MB ", end='')
                
            # Scramble data, this converts the numpy dataset into Tensors
            train, valid = dataset.get_training_data(dataset.validation_split, pin_memory=True)
            
            # Pin memory
            train_data = []
            if self.pin_memory == True:
                for data in tqdm(train, desc="Pinning", bar_format=bar_format):
                    X    : torch.Tensor = data["X"].float().to(self.device)
                    Y    : torch.Tensor = data["y"].float().to(self.device)
                    train_data.append({ "X": X, "y": Y })
            else:
                # Putting the data in a list seems to be slightly faster than
                # iterating over the dataset directly for some reason
                for data in tqdm(train, desc="Preparing", bar_format=bar_format):
                    X    : torch.Tensor = data["X"]
                    Y    : torch.Tensor = data["y"]
                    train_data.append({ "X": X, "y": Y })
            
            # Train model
            module.train(True)
            train_loss = 0.0
            train_err = 0.0
            train_acc = 0.0
            train_err_max = 0.0
                    
            train_progress = tqdm(train_data, bar_format=bar_format)
            for train_iter, data in enumerate(train_progress):
                X, Y = data["X"], data["y"]

                if data_callback != None:
                    data_callback(**{
                        "iter": train_iter,
                        "x": X,
                        "y": Y,
                    })

                # zero the parameter gradients
                y_hat, loss = self.single_train(X, Y, loss_func, optimizer)

                # print statistics
                train_loss += loss.item()
                if math.isnan(train_loss):
                    self.print_debug_nan(X, Y, optimizer)
        
                err_vec = (torch.abs(y_hat - Y))
                train_err += err_vec.mean().item()
                train_err_max = max(train_err_max, self.scale_output(err_vec.max().item(), column=dataset.target, is_delta=True))
                
                if self.scale_output(err_vec.max().item(), column=dataset.target, is_delta=True) > 8:
                    print("ERR_VEC", err_vec)
                    print("X", self.scale_output(X[0,:,0]))
                    print("Y", self.scale_output(Y[0,:], is_delta=True))
                    print("Y_HAT", self.scale_output(y_hat[0,:], is_delta=True))
                    
                    raise ""
                
                # b, n, f
                train_acc += (torch.sign(y_hat * Y) > 0).sum().item() / y_hat.numel()
                
                disp_loss = train_loss / (train_iter + 1)
                disp_acc = train_acc / (train_iter + 1)
                disp_err = self.scale_output(train_err / (train_iter + 1), column=dataset.target, is_delta=True) # accurate if loss < 1
                
                if iter_callback != None:
                    iter_callback(**{
                        "iter": train_iter,
                        "y_hat": y_hat,
                        "err": disp_err,
                        "err_max": train_err_max,
                    })
                    
                train_progress.set_postfix({
                    # "loss": format_loss(disp_loss),
                    "acc": format_loss(disp_acc),
                    "err($)": format_loss(disp_err),
                    "err_max($)": format_loss(train_err_max),
                    }, refresh=False)
                
            # Validate results
            module.eval()
            valid_loss = 0.0
            valid_err = 0.0
            valid_acc = 0.0
            valid_progress = tqdm(valid, bar_format=bar_format)
            with torch.no_grad():
                for valid_iter, data in enumerate(valid_progress):
                    if self.pin_memory:
                        X    : torch.Tensor = data["X"].float().to(self.device)
                        Y    : torch.Tensor = data["y"].float().to(self.device)
                    else:
                        X    : torch.Tensor = data["X"]
                        Y    : torch.Tensor = data["y"]
                    
                    y_hat, loss = self.single_infer(X, Y, loss_func)

                    # print statistics
                    valid_loss += loss.item()
                    if math.isnan(valid_loss):
                        print("\nSaving failed model separately for debugging: ")
                        self.save("ckpt/fail_" + self.get_filename())
                        raise ArithmeticError("Failed training, val_loss = NaN")
            
                    valid_err += (torch.abs(y_hat - Y)).mean().item()
                    valid_acc += (torch.sign(y_hat * Y) > 0).sum().item() / y_hat.numel()
                
                    # val_loss = valid_loss / (valid_iter + 1)
                    val_err = self.scale_output(valid_err / (valid_iter + 1), column=dataset.target, is_delta=True)
                    val_acc = valid_acc / (valid_iter + 1)
                    
                    valid_progress.set_postfix({
                        # "val_loss": format_loss(val_loss),
                        "val_acc": format_loss(val_acc),
                        "val_err($)": format_loss(val_err) }, refresh=False)
                
            self.runtime['epoch'] += 1
                
            print()
            
        print(f"Done in user: {time.time() - start_time:.2f}s; sys: {time.process_time() - pstart_time:.2f}s.\n")
        self.runtime['loss'] = train_loss / train_iter
        self.runtime['val_loss'] = valid_loss / valid_iter
        self.optimizer_state = None if optimizer is None else optimizer.state_dict()
            
    def save(self, filename=None):
        """Saves the model to a checkpoint file. If no filename is given, defaults to ckpt/{StandardModel.get_filename()}.
        
        The model must be initialized. Models can be initialized by one of two methods: loading from file with self.load(),
        or training a new model with self.standard_train(dataset).

        Args:
            filename (str, optional): The path to save the checkpoint file to. Defaults to ckpt/{StandardModel.get_filename()}.

        Raises:
            RuntimeError: If there is no runtime data from an initialized model (see above), raises a RuntimeError.
        """
        
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        
        print("> Saving model: ")
        
        if self.runtime == None:
            raise RuntimeError("Cannot save model without running it first.")
        
        # self.runtime is initialized upon self.standard_train(), or upon self.load().
        ckpt = {
            'epoch': self.runtime["epoch"],
            'model_state': self.module.state_dict(),
            'optimizer_state': self.optimizer_state,
            'loss': self.runtime["loss"],
            'val_loss': self.runtime["val_loss"],
            
            'columns': self.runtime['columns'],
            'mean_': self.runtime['mean_'],
            'var_': self.runtime['var_'],
            'scale_': self.runtime['scale_'],
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
        
        # has saved scaler data
        if 'mean_' in ckpt:
            self.runtime = {
                'epoch': ckpt['epoch'],
                'loss': ckpt['loss'],
                'val_loss': ckpt['val_loss'],
                
                'columns': ckpt['columns'],
                'mean_': ckpt['mean_'],
                'var_': ckpt['var_'],
                'scale_': ckpt['scale_'],
            }
            
            if not 'columns' in ckpt:
                print("> !!! Model has no columns, saved model may not match.")
            elif self.runtime['columns'] != self.conf.column_names:
                raise ValueError(f"Model is invalid; Columns are different from saved model. Old: {self.runtime['columns']}, New: {self.conf.column_names}.")
            
            # Set scaler values
            self.scaler.mean_ = self.runtime['mean_']
            self.scaler.var_ = self.runtime['var_']
            self.scaler.scale_ = self.runtime['scale_']
        # does not have scaled scaler data (legacy), will fit in standard_train.
        else:
            self.runtime = {
                'epoch': ckpt['epoch'],
                'loss': ckpt['loss'],
                'val_loss': ckpt['val_loss'],
            }
        
        print(self.runtime)
        print()
        
    def print_debug_nan(self, X, Y, optimizer):
        print(">>> IN/OUT: ")
        print(X, Y)
        print()
        print()
        try:
            with torch.autograd.detect_anomaly():
                y_hat = self.module.forward(X)
                torch.mean(y_hat).backward()
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
        
import deepspeed
        
class DeepspeedModel(PytorchModel):
    def __init__(self, module: nn.Module, model_json, device=None, verbosity=1):
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
        
    def single_train(self, X, Y, loss_func, optimizer):
        y_hat = self.model_engine(X)
        loss = loss_func.forward(y_hat, Y)
        
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        return y_hat, loss
        
    def single_infer(self, X, Y, loss_func):
        y_hat = self.model_engine(X)
        loss = loss_func.forward(y_hat, Y)
        
        return y_hat, loss
            
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