import numpy as np
import random
import copy
from collections import namedtuple, deque

from modelb1 import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.99               # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 5e-5         # learning rate of the actor 
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1        # Learn every
NOISE_DECAY = 99.99e-2     # Noise decay multiplier
STOP_NOISE = 1e6        # Timesteps to stop adding noise
SOFT_UPDATE_EVERY = 4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    
    def __init__(self, state_size, action_size, n_agents, random_seed, agent_id):
        self.state_size = state_size
        self.action_size = action_size
        
        self.id = agent_id
        self.t_step = 0
        
        self.noice_decay = NOISE_DECAY
        self.stop_noice = STOP_NOISE
        
        # Construct Actor networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        # Construct Critic networks 
        self.critic_local = Critic(state_size, action_size, n_agents, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, n_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
            
    def step(self, memory, agents):
        self.t_step += 1
        if len(memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            experiences = memory.sample()
            self.learn(experiences, GAMMA, agents)        

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise and self.t_step <= self.stop_noice:
            noi = self.noise.sample()
            action += noi*self.noice_decay
            self.noice_decay *= self.noice_decay
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, all_agents):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_list, actions_list, rewards, next_states_list, dones = experiences
        agent_id = torch.tensor([self.id]).to(device)
        
        next_states_tensor = torch.cat(next_states_list, dim=1).to(device)
        states_tensor = torch.cat(states_list, dim=1).to(device)
        actions_tensor = torch.cat(actions_list, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = [agent.actor_target(states) for agent, states in zip(all_agents, next_states_list)]     
#         next_actions = [self.actor_target(states) for states in next_states_list] 
        next_actions_tensor = torch.cat(next_actions, dim=1).to(device)
        Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)        
        # Compute Q targets for current states (y_i)
#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))  
        Q_targets = rewards.index_select(1, agent_id) + (gamma * Q_targets_next * (1 - dones.index_select(1, agent_id)))
        # Compute critic loss
        Q_expected = self.critic_local(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions
#         actions_pred = [self.actor_local(states) for states in states_list]  
#         actions_pred = [agent.actor_local(states) if i==self.id else agent.actor_local(states).detach() 
#                         for i, (agent, states) in enumerate(zip(all_agents, states_list))]
        actions_pred = [agent.actor_local(states) for agent, states in zip(all_agents, states_list)]
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
#         if self.t_step % SOFT_UPDATE_EVERY == 0:
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class MADDGP:
    def __init__(self, state_size, action_size, n_agents, random_seed):
        self.n_agents = n_agents
        self.action_size = action_size
        self.agents = [Agent(state_size, action_size, n_agents, random_seed, i) for i in range(self.n_agents)]
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, n_agents)
        
    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        for agent in self.agents:
            agent.step(self.memory, self.agents)
    
    def act(self, states, add_noise=True):
        actions = np.zeros([self.n_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions
    
    def reset(self):        
        for agent in self.agents:
            agent.reset()
            
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, n_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
        self.n_agents = n_agents
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states_list = [torch.from_numpy(np.vstack([e.states[i] for e in experiences if e is not None]))
                       .float().to(device) for i in range(self.n_agents)]
        actions_list = [torch.from_numpy(np.vstack([e.actions[i] for e in experiences if e is not None]))
                        .float().to(device) for i in range(self.n_agents)]
        next_states_list = [torch.from_numpy(np.vstack([e.next_states[i] for e in experiences if e is not None]))
                            .float().to(device) for i in range(self.n_agents)]            
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)        
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_list, actions_list, rewards, next_states_list, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)