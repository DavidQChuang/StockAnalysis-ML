from torch import nn
import torch

from traders.Common import TraderConfig

class ComputeRewards(nn.Module):
    def __init__(self, conf: TraderConfig, loss_func, policy_net, target_net):
        self.conf = conf
        self.loss_func = loss_func
        self.policy_net = policy_net
        self.target_net = target_net
    
    def forward(self, batch: torch.Tensor):
        device = batch.device

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None])
        # (b, n_observations)
        state_batch = torch.stack(batch.state)
        # (b, 1)
        action_batch = torch.stack(batch.action)
        # (b, 1)
        reward_batch = torch.stack(batch.reward)

        # -- Compute Q(s_t, a)
        # The model computes Q(s_t), then we select the reward values Q(s_t, a)
        # corresponding to the prevoously taken action.
        # INPUT: state_batch:                   (b, o)
        #        policy_net(state_batch):       (b, a)
        #        gather(1, action_batch):       (b, 1)
        action_rewards = self.policy_net(state_batch).gather(1, action_batch)

        # -- Compute V(s_{t+1}) for all next states.
        # Use "older" target_net to select the best reward with max(1)[0].
        # This is only computed for non-final states.
        # Final states are set to the last reward.
        next_state_values = (torch.zeros((self.conf.batch_size,1)) - 250).to(device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(-1)
            
        # Compute the expected Q values
        expected_action_rewards = (next_state_values * self.conf.gamma) + reward_batch
        
        # Compute Huber loss
        loss = self.loss_func(action_rewards, expected_action_rewards)
        
        return loss