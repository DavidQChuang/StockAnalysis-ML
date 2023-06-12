from torch import nn
import torch.nn.functional as F

class LinearDQN(nn.Module):
    def __init__(self, n_observations, n_actions, hl_size):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hl_size)
        self.layer2 = nn.Linear(hl_size, hl_size)
        self.layer3 = nn.Linear(hl_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)