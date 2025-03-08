# PPO and GRPO Agents

This is a personal and experimental (for fun 😊)repository implements Proximal Policy Optimization (PPO) to train an agent on CartPole-v1 using PyTorch and Gymnasium. 

📊Training Insights

The mean reward plot shows rapid learning, reaching near-optimal performance (500 reward) within ~100 episodes.
The confidence interval plot (below) highlights the reward variance across the 5 training runs of 200 episodes.

![training](plots/confidence_interval.png)




🚀How to Run

Install dependencies:

`pip install -r requirements.txt`

Train the PPO agent:

`python agents.py`

Generate and visualize results:

`python plot.py`


