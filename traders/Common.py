
from collections import deque, namedtuple
from dataclasses import dataclass
import inspect
import math
import random
import time
import numpy as np

from tqdm import tqdm

from datasets.Common import DatasetConfig, TimeSeriesDataset
from models.Common import StandardModel, get_bar_format

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from traders.LinearDQN import LinearDQN
from traders.TradingDataset import TradingDataset
from traders.TradingSimulation import TradingSimulation

@dataclass
class TraderConfig:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
    
    # Trainer params
    batch_size  : int   = 128
    gamma       : float = 0.99
    eps_start   : float = 0.9
    eps_end     : float = 0.05
    eps_decay   : float = 1000
    tau         : float = 0.005
    lr          : float = 1e-4
    
    episodes      : int   = 256
    
    hidden_layer_size: int = 2048
    
    # Economy
    starting_money: float = 5000.
    
    @property
    def model_filename(self):
        return "%d" % (
            self.hidden_layer_size)
    
Transition = namedtuple('Transition',
            ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class StandardTrader:
    def __init__(self, trader_json, device=None):
        self.conf = conf = TraderConfig.from_dict(trader_json)
        self.device = torch.device(device or "cpu")
        
        self.runtime = None
        self.starting_money = conf.starting_money
        self.money = conf.starting_money
        
        self.volume = 0
        
        self.optimizer_state = None
        self.policy_net_state = None
        self.target_net_state = None
        
    def get_model_name(self):
        return self.__class__.__name__
    
    def single_train_action(self, policy_net, state, steps):
        conf = self.conf
        device = self.device
        
        eps = conf.eps_end + (conf.eps_start - conf.eps_end) * math.exp(-1 * steps / conf.eps_decay)
        steps += 1
        
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                # Pick the action with the greatest reward
                predict_rewards = policy_net(state)
                # max(..)[1] is the 
                return predict_rewards.max(0)[1].unsqueeze(-1)
        else:
            p = random.random()
            if p < 0.10:
                value = 1
            elif p < 0.20:
                value = 2
            elif p < 0.25:
                value = 3
            else:
                value = 0
             
            return torch.IntTensor([value]).to(device)
        
    def single_train(self, memory: ReplayMemory, policy_net: nn.Module, target_net: nn.Module, optimizer):
        conf = self.conf
        device = self.device
        
        # b: batch_size; o: observations; a: actions
        if len(memory) < conf.batch_size:
            return
        
        # Get batch of transitions from memory
        transitions = memory.sample(conf.batch_size)
        # Convert list of Transitions into Transition storing batched data
        # (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*transitions))

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
        if action_batch.dtype != torch.int64:
            print(action_batch)
            raise Exception()  
        action_rewards = policy_net(state_batch).gather(1, action_batch)

        # -- Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((conf.batch_size,1)).to(device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].unsqueeze(-1)
        # Compute the expected Q values
        expected_action_rewards = (next_state_values * conf.gamma) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(action_rewards, expected_action_rewards)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        
    def get_state(self, trade_sim: TradingSimulation, datapoint, device):
        """Returns the current state of the DQN.
        Concatenates real datapoints first, then inferred future datapoints,
        then account value.

        Args:
            trade_sim (_type_): _description_
            datapoint (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        price_state = torch.Tensor(datapoint["real"]).to(device)
        infer_state = torch.Tensor(datapoint["inference"]).to(device)
        # Get account state from current price and future prediced price
        account_state = torch.Tensor(trade_sim.state(datapoint["real"][-1], datapoint["inference"][0])).to(device)
        return torch.cat([price_state, infer_state, account_state])
        
    def standard_train(self, model: StandardModel, dataset: TimeSeriesDataset):
        conf = self.conf
        device = torch.device(self.device)
        use_cuda = self.device != None and self.device != 'cpu'
        
        # Log trader setup
        print(f'> Trader config:')
        print(f'{conf}')
        print()
        print(f'> Training trader {self.get_model_name()}.')
        
        trade_sim = TradingSimulation(conf.starting_money)
        scaled_dataset = model.scale_dataset(dataset, fit=True)
        scaled_dataset = TradingDataset(scaled_dataset.df, model)
        
        # Set up DQN
        # Inputs: stock prices up to current time, future predicted prices, and current account state
        n_observations = model.conf.seq_len + model.conf.out_seq_len + TradingSimulation.state_size()
        # Outputs: nothing, buy, sell
        n_actions = TradingSimulation.action_count()
        
        policy_net = torch.compile(LinearDQN(n_observations, n_actions, self.conf.hidden_layer_size).to(device))
        target_net = torch.compile(LinearDQN(n_observations, n_actions, self.conf.hidden_layer_size).to(device))
        
        if self.policy_net_state != None:
            policy_net.load_state_dict(self.policy_net_state)
        if self.target_net_state != None:
            target_net.load_state_dict(self.target_net_state)

        optimizer = self.get_optimizer(policy_net, conf)
        memory = ReplayMemory(10000)
        
        # Runtime logging stuff
        first_run = self.runtime == None
        if first_run:
            self.runtime = {
                'episode': 0,
                'steps': 0
            }
        lifetime_episodes = self.runtime['episode'] + conf.episodes
            
        # -- Train multiple episodes
        # Time realtime and cpu time
        start_time = time.time()
        pstart_time = time.process_time()
        bar_format = get_bar_format(len(scaled_dataset), conf.batch_size)
        
        batch_count = math.ceil(len(scaled_dataset) / conf.batch_size)
        
        # Global action steps for epsilon decay
        steps = 0
        
        verybad, bad, good, goodish, verygood = 0, 0, 0, 0, 0
        for e in range(conf.episodes):
            addl_desc = '' if first_run else f'; Total episodes: {self.runtime["episode"] + 1}/{lifetime_episodes}'
            print(f"Episode {e+1}/{conf.episodes}{addl_desc}")
            if use_cuda:
                print(f"GPU: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                
            # -- Initialize new state
            trade_sim = TradingSimulation(conf.starting_money)
            
            # Get random contiguous subset of dataset
            # Since network is trained on delta, the last step of the batch 
            # is taken out of the direct training loop
            batch = random.randint(0, batch_count - 1)
            idx_range = range(
                batch * conf.batch_size,
                min((batch + 1) * conf.batch_size, len(scaled_dataset)) - 1)
            
            datapoint = scaled_dataset[idx_range.start]
            start_price = model.scale_output(datapoint["real"][-1])
            state = self.get_state(trade_sim, datapoint, device)
            
            # -- Run the simulation
            train_progress = tqdm(idx_range, bar_format=bar_format)
            avg_profit = 0
            avg_delta = 0
            avg_mkt_profit = 0
            avg_mkt_delta = 0
            for idx in train_progress:
                # Step with action, then compute next valuation
                action = self.single_train_action(policy_net, state, steps)
                
                datapoint = scaled_dataset[idx]
                next_datapoint = scaled_dataset[idx + 1]
                
                # Stock pricing
                curr_price = model.scale_output(datapoint["real"][-1])
                next_price = model.scale_output(next_datapoint["real"][-1])
                # Value of the simulated stock account
                curr_value = trade_sim.step(action, curr_price)
                next_value = trade_sim.valuation(next_price) 
                
                # Value of the market (if you had invested the starting money and kept it in the market the whole time)
                market_value = trade_sim.market_valuation(start_price, curr_price)
                next_market_value = trade_sim.market_valuation(start_price, next_price)
                
                # Reward is delta liquid valuation
                #reward = (next_value - curr_value) # - (next_market_value - market_value)
                profit = (next_value - conf.starting_money)
                delta = (next_value - curr_value)
                mkt_profit = (next_market_value - conf.starting_money)
                mkt_delta = (next_market_value - market_value)
                reward = trade_sim.scale_delta(profit + (next_value - next_market_value))

                avg_profit += profit
                avg_delta += delta
                avg_mkt_profit += mkt_profit
                avg_mkt_delta += mkt_delta

                # -- Step state
                # End simulation prematurely if we lost 10% of our money
                if next_value < next_market_value * 0.9:
                    next_state = None
                    print("\nAbort.")
                else:
                    next_state = self.get_state(trade_sim, next_datapoint, device)

                # Store the transition in memory
                memory.push(state, action, next_state, torch.tensor([reward]).to(device))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.single_train(memory, policy_net, target_net, optimizer)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*conf.tau + target_net_state_dict[key]*(1-conf.tau)
                target_net.load_state_dict(target_net_state_dict)
                
                if next_state is None:
                    break
                else:
                    train_progress.set_postfix({"val": f'${next_value:.2f}',
                                            "mkt_val":f'${next_market_value:.2f}',
                                            "reward": f'${reward:.2f}',
                                            "delta": f'${next_value-curr_value:.2f}',
                                            "profit": f'${next_value-conf.starting_money:.2f}'},
                                            refresh=False)

            self.runtime['episode'] += 1

            # good market
            if avg_mkt_profit > 0:
                # good trade
                if avg_profit > 0:
                    good += 1
                else:
                    verybad += 1
            # bad market
            else:
                # good trade
                if avg_profit > 0:
                    verygood += 1
                elif avg_profit > avg_mkt_profit:
                    goodish += 1
                else:
                    bad += 1
            
            iters = len(idx_range)
            print(f"avg_mkt_profit: {avg_mkt_profit/iters:.2f}$; avg_mkt_delta: {avg_mkt_delta/iters:.2f}$")
            print(f"avg_profit: {avg_profit/iters:.2f}$; avg_delta: {avg_delta/iters:.2f}$")
            print(f"avg_profit diff: {(avg_profit-avg_mkt_profit)/iters:.2f}$; avg_delta diff: {(avg_delta - avg_mkt_delta)/iters:.2f}$")
            print(f"very bad(+/-): {verybad}, bad(-/-): {bad}, good(+/+): {good}, goodish(-<-): {goodish}, very good(-/+): {verygood}")
            print(f"user: {time.time() - start_time:.2f}s; sys: {time.process_time() - pstart_time:.2f}s.")
            print()
            
        print(f"Done in user: {time.time() - start_time:.2f}s; sys: {time.process_time() - pstart_time:.2f}s.\n")
        self.runtime['steps'] = steps
        self.policy_net_state = policy_net.state_dict()
        self.target_net_state = target_net.state_dict()
        self.optimizer_state = optimizer.state_dict()
            
    def get_optimizer(self, policy_net, conf):
        optimizer = optim.AdamW(policy_net.parameters(), lr=conf.lr, amsgrad=True)
        
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        
        return optimizer
    
    def get_filename(self):
        filename = self.conf.model_filename
        
        return f"Trader-{filename}.ckpt"
    
    def save(self, filename=None):
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        
        print("> Saving trader: ")
        
        if self.runtime == None:
            raise Exception("Cannot save model without running it first.")
        
        ckpt = {
            'episode': self.runtime["episode"],
            'steps': self.runtime["steps"],
            'policy_net_state': self.policy_net_state,
            'target_net_state': self.target_net_state,
            'optimizer_state': self.optimizer_state
        }
        print({
            'episode': ckpt['episode']
            })
        print()
        
        torch.save(ckpt, filename)
            
    def load(self, filename=None):
        filename = filename if filename != None else f"ckpt/{self.get_filename()}"
        
        print("> Loading trader: ")
        
        ckpt = torch.load(filename)
        
        self.policy_net_state = ckpt['policy_net_state']
        self.target_net_state = ckpt['target_net_state']
        self.optimizer_state = ckpt['optimizer_state']
        
        self.runtime = {
            'episode': ckpt['episode'],
            'steps': ckpt["steps"]
        }
        
        print(self.runtime)
        print()