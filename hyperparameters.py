
# Hyperparameters for the PPO algorithm
# The hyperparameters are based on the original paper: https://arxiv.org/abs/1707.06347
HYPERPARAMETERS_PPO = {
    "GAMMA": 0.99,
    "LR": 0.002,
    "EPS_CLIP": 0.2,  # Clipping range for PPO
    "K_EPOCHS": 4,  # Number of optimization epochs
    "T_MAX": 2000,  # Maximum training steps per episode
    "BATCH_SIZE": 32,
    "HIDDEN_DIM": 128
}


# The key differences in the hyperparameters for GRPO are the KL target and KL beta
# The KL target is the target KL divergence for adaptive adjustment
# The KL beta is the initial weight of the KL penalty term


HYPERPARAMETERS_GRPO = {
    'LR': 2e-4,  # Learning rate
    'EPS_CLIP': 0.2,  # Clipping range (PPO) -- replaced in GRPO by KL regularization
    'GAMMA': 0.99,  # Discount factor
    'LAMBDA': 0.95,  # GAE parameter
    'K_EPOCHS': 20,  # Update epochs per optimization step
    'KL_TARGET': 0.05,  # Target KL divergence for adaptive adjustment
    'KL_BETA': 0.05,  # Initial weight of KL penalty term
    'LR_DECAY': 0.9,  # Decay factor for learning rate if KL exceeds threshold
    'T_MAX': 2000  # Maximum training steps per episode
}