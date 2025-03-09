import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from hyperparameters import HYPERPARAMETERS_PPO, HYPERPARAMETERS_GRPO
import os

# Define the Actor-Critic Network for both PPO 
class ActorCriticPPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HYPERPARAMETERS_PPO['HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(HYPERPARAMETERS_PPO['HIDDEN_DIM'], action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HYPERPARAMETERS_PPO['HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(HYPERPARAMETERS_PPO['HIDDEN_DIM'], 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

# PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=HYPERPARAMETERS_PPO['LR']) # Optimizer
        self.memory = []  # Memory to store transitions
        self.eps_clip = HYPERPARAMETERS_PPO['EPS_CLIP'] # PPO clip for ratio of new policy to old policy
        self.gamma = HYPERPARAMETERS_PPO['GAMMA'] # Discount factor

    def select_action(self, state):
        if isinstance(state, tuple): 
            state = np.array(state)  # Convert tuple to numpy array if needed
        if len(state) == 0:
            raise ValueError("Error: Received an empty state.")
        state = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_transition(self, transition):
        self.memory.append(transition)

    def optimize(self):
        states, actions, log_probs_old, rewards, dones = zip(*self.memory)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs_old = torch.stack(log_probs_old).detach()  # Ensure it's detached from old graph
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute discounted rewards
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(HYPERPARAMETERS_PPO['K_EPOCHS']):
            torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            log_probs_new = dist.log_prob(actions)

            # Compute advantage without detaching
            advantages = returns - state_values.squeeze()

            # PPO ratio
            ratio = torch.exp(log_probs_new - log_probs_old)

            # Clipped ratio
            clipped_ratio = torch.clamp(ratio, min=1 - self.eps_clip, max=1 + self.eps_clip)

            # Compute policy loss
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Compute value loss without detaching state_values
            value_loss = nn.MSELoss()(state_values.squeeze(), returns)

            # Combine losses
            loss = policy_loss + value_loss

            # Debugging print statements
            #print(f"Debug: Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

            # Update policy parameters
            self.optimizer.zero_grad()
            loss.backward()  # No retain_graph=True
            self.optimizer.step()

        # Clear memory
        self.memory.clear()

# Define Actor-Critic Network for GRPO : slightly different from PPO 
class ActorCriticGRPO(nn.Module):
    """
    Actor-Critic network for GRPO.
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCriticGRPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

class GRPO:
    def __init__(self, state_dim, action_dim):
        """
        Initialize the GRPO agent with the Actor-Critic network, optimizer, memory, and hyperparameters.
        """
        self.policy = ActorCriticGRPO(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=HYPERPARAMETERS_GRPO['LR'])
        self.memory = []  
        self.gamma = HYPERPARAMETERS_GRPO['GAMMA']
        self.lam = HYPERPARAMETERS_GRPO['LAMBDA']
        self.kl_beta = HYPERPARAMETERS_GRPO['KL_BETA']
        self.action_dim = action_dim

    def select_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # ✅ FIXED: Return action_probs so it can be stored in memory for KL computation
        return action.item(), log_prob, action_probs.detach()

    def store_transition(self, transition):
        self.memory.append(transition)

    def optimize(self):
        """
        Update the policy using stored memory transitions.
        """
    
        states, actions, log_probs_old, rewards, dones, old_action_probs = zip(*self.memory)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs_old = torch.stack(log_probs_old).detach()
        old_action_probs = torch.stack(old_action_probs).detach()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # GAE computation
        advantages = []
        returns = []
        R = 0
        A = 0
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(old_action_probs)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
            delta = r + self.gamma * v.max() - R  # Ensure proper advantage computation
            A = delta + self.gamma * self.lam * A * (1 - d)
            advantages.insert(0, A)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        for _ in range(HYPERPARAMETERS_GRPO['K_EPOCHS']):
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            log_probs_new = dist.log_prob(actions)

            # KL divergence computation
            kl_divergence = torch.distributions.kl.kl_divergence(Categorical(old_action_probs), Categorical(action_probs))

            # ratio for policy loss
            ratio = torch.exp(log_probs_new - log_probs_old)

            # Policy loss with KL penalty
            policy_loss = -torch.mean(torch.min(ratio * advantages, self.kl_beta * kl_divergence))

            # Value loss
            value_loss = nn.MSELoss()(state_values.squeeze(), returns)

            # Total loss with KL penalty term
            loss = policy_loss + value_loss + self.kl_beta * kl_divergence.mean()

            # Adjust KL beta dynamically
            if kl_divergence.mean().item() > HYPERPARAMETERS_GRPO['KL_TARGET']:
                self.kl_beta *= 1.5  # Increase penalty
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= HYPERPARAMETERS_GRPO['LR_DECAY']  # Reduce learning rate
            elif kl_divergence.mean().item() < HYPERPARAMETERS_GRPO['KL_TARGET'] / 2:
                self.kl_beta /= 1.5  # Reduce penalty

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()


if __name__ == "__main__":
    # Run multiple seeds to analyze variance in rewards
    env_name = "CartPole-v1"  # Other options: "LunarLander-v2", "Pendulum-v0"
    num_seeds = 5
    num_episodes = 200

    seeds = np.random.randint(0, 123456, num_seeds)

    print(f"Training PPO agent on {env_name} environment.")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Initialize PPO agent
        ppo_agent = PPO(state_dim, action_dim)
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            # Fix for Gym version compatibility:
            if isinstance(state, tuple):
                state, _ = state  # Unpack (state, info) when gym returns a tuple
            state = np.array(state, dtype=np.float32)  # Ensure it's a proper NumPy array
            episode_reward = 0
            for t in range(HYPERPARAMETERS_PPO['T_MAX']):
                action, log_prob = ppo_agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # Ensure compatibility with new Gym versions
                ppo_agent.store_transition((state, action, log_prob, reward, done))
                state = next_state
                episode_reward += reward
                if done:
                    break
            
            ppo_agent.optimize()
            if (episode + 1) % 10 == 0:
                print(f"Seed {seed} - Episode {episode+1}: Reward = {episode_reward}")
                rewards.append(episode_reward)
            

        # Save rewards for this seed
        os.makedirs("results_ppo", exist_ok=True)
        np.savetxt(f"results/rewards_{env_name}_seed{seed}.csv", rewards, delimiter=",")
        env.close()

    print("Training complete.")

    num_episodes =1000 # Increase number of episodes for GRPO training
    
    print(f"Training GRPO agent on {env_name} environment.")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Initialize GRPO agent
        grpo_agent = GRPO(state_dim, action_dim)
        rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            # Fix for Gym version compatibility:
            if isinstance(state, tuple):
                state, _ = state  # Unpack (state, info) when gym returns a tuple
            state = np.array(state, dtype=np.float32)  # Ensure it's a proper NumPy array
            episode_reward = 0
            for t in range(HYPERPARAMETERS_GRPO['T_MAX']):
                action, log_prob, action_probs = grpo_agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # Ensure compatibility with new Gym versions
                grpo_agent.store_transition((state, action, log_prob, reward, done, action_probs))
                state = next_state
                episode_reward += reward
                if done:
                    break
            
            grpo_agent.optimize()
            if (episode + 1) % 10 == 0:
                print(f"Seed {seed} - Episode {episode+1}: Reward = {episode_reward}")
                rewards.append(episode_reward)

        # Save rewards for this seed
        os.makedirs("results_grpo", exist_ok=True)
        np.savetxt(f"results_grpo/rewards_{env_name}_seed{seed}.csv", rewards, delimiter=",")
        env.close()

    print("Training complete.")

# The code snippet above trains both the PPO and GRPO agents on the CartPole environment using the same hyperparameters. 
# The rewards for each seed are saved to separate CSV files for analysis.
