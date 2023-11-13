from torch import nn
import torch.nn.functional as F

class LinearDQN(nn.Module):
    def __init__(self, n_observations, n_actions, hl_size):
        super().__init__()
        
        self.hl_size = hl_size
        
        self.layer1 = nn.Linear(n_observations, hl_size)
        self.layer2 = nn.Linear(hl_size, hl_size)
        self.lstm = nn.LSTM(hl_size, hl_size, batch_first=True)
        self.layer3 = nn.Linear(hl_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        print("ASFASFA", x.shape)
        x = F.relu(self.layer2(x))
        print("ASFASFSAFA",x.shape)
        x, hx = self.lstm(x.reshape([x.shape[0], 1, self.hl_size]))
        x = F.relu(x.reshape([self.hl_size]))
        return self.layer3(x)