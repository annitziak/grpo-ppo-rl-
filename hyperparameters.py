
# Hyperparameters for the PPO algorithm
# The hyperparameters are based on the original paper: https://arxiv.org/abs/1707.06347
HYPERPARAMETERS = {
    "GAMMA": 0.99,
    "LR": 0.002,
    "EPS_CLIP": 0.2,  # Clipping range for PPO
    "K_EPOCHS": 4,  # Number of optimization epochs
    "T_MAX": 2000,  # Maximum training steps per episode
    "BATCH_SIZE": 32,
    "HIDDEN_DIM": 128
}
