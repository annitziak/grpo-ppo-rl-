import os
import numpy as np
import matplotlib.pyplot as plt

def plot(directory):
    """
    Plot the reward curves from all the reward files in the specified directory.
    
    Args:
    - directory (str): The path to the directory containing the reward files.
    """
    files = [f for f in os.listdir(directory) if f.startswith("rewards_") and f.endswith(".csv")]
    
    all_rewards = []
    for file in files:
        filepath = os.path.join(directory, file)
        rewards = np.loadtxt(filepath, delimiter=",")
        all_rewards.append(rewards)
    
    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    episodes = np.arange(len(mean_rewards))
    
    # since the values are over 200 episodes overy 10 
    xticks = np.linspace(10, 200, num=len(episodes))


    # Create results directory for saving plots
    os.makedirs("plots", exist_ok=True)
    
    # Plot all individual reward curves
    plt.figure(figsize=(10, 6))
    for rewards in all_rewards:
        plt.plot(episodes, rewards, alpha=0.5)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("All Reward Curves")
    plt.xticks(np.arange(len(episodes)), np.arange(10, 201, 10))
    plt.grid()
    plt.savefig("plots/all_rewards.png")
    plt.show()
    
    # Plot mean reward over all steps
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards, label="Mean Reward", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Mean Reward Over All Steps")
    plt.xticks(np.arange(len(episodes)), np.arange(10, 201, 10))
    plt.legend()
    plt.grid()
    plt.savefig("plots/mean_rewards.png")
    plt.show()
    
    # Plot confidence interval (mean ± std deviation)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards, label="Mean Reward", color='blue')
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.3, label="Confidence Interval")
    plt.xticks(np.arange(len(episodes)), np.arange(10, 201, 10))
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Mean Reward with Confidence Interval")
    plt.legend()
    plt.grid()
    plt.savefig("plots/confidence_interval.png")
    plt.show()

if __name__ == "__main__":
    plot("results_ppo")
# The `plot` function reads all the reward files in the specified directory, calculates the mean and standard deviation of the rewards at each episode, and then plots the following: