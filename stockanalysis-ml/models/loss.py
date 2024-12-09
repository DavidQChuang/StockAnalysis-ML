from functools import reduce
from torch import Tensor
import torch
import torch.functional as F

class MADLoss:
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # loss = input * target
        loss = torch.sign(prediction * target) * torch.abs(target)
        return -loss.mean()
    
class CombinedLoss:
    def __init__(self, funcs) -> None:
        self.funcs = funcs
        
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        result1 = self.funcs[0].forward(prediction, target)
        return reduce(lambda acc, func: acc + func.forward(prediction, target), self.funcs[1:], result1)
        