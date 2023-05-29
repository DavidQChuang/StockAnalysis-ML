from dataclasses import dataclass, field

@dataclass(frozen=True)
class StandardConfig:
    id: str
    
    # Trainer params
    loss_name       : str   = "mean_squared_error"
    optimizer_name  : str   = "adam"
    batch_size      : int   = 64
    epochs          : int   = 15
    
    # General NN architecture params
    hidden_layer_size   : int = 50
    dropout_rate        : int = 0.3
    
    # Model I/O parameters
    in_window           : int = 24
    out_window          : int = 5
    
    @property
    def model_filename(self):
        return "%s%d_%d+%d" % (
            id, self.hidden_layer_size, self.in_window, self.out_window)