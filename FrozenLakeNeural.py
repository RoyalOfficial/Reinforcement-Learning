import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

size = 16
custom_map = []

# Save the map if it doesn't exist
map_path = "frozen_map.npy"
if os.path.exists(map_path):
    print("Loading saved map...")
    custom_map = np.load(map_path, allow_pickle=True).tolist()
else:
    print("Generating new map...")
    custom_map = []
    for i in range(size):
        row = ""
        for j in range(size):
            if i == 0 and j == 0:
                row += "S"
            elif i == size-1 and j == size-1:
                row += "G"
            else:
                row += random.choice(["F", "F", "F", "H"])
        custom_map.append(row)
    np.save(map_path, np.array(custom_map, dtype=object))

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode = "human")

#neural net
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
num_states = env.observation_space.n
num_actions = env.action_space.n

#Try to load existing qnet
if os.path.exists("frozenlake_qnet.pth"):
    print("Loading saved Q-network...")
    q_net = QNetwork(num_states, num_actions)
    q_net.load_state_dict(torch.load("frozenlake_qnet.pth", weights_only=True))
    q_net.eval()
else:
    print("No saved Q-network found. Creating new model...")
    q_net = QNetwork(num_states, num_actions)

optimizer = optim.Adam(q_net.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

#Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 5000

#One-hot encoding
def one_hot(state, state_size):
    vec = np.zeros(state_size)
    vec[state] = 1
    return vec

#Training loop
for epsiode in range(episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        state_vec = torch.tensor(one_hot(state, num_states), dtype=torch.float32)
        
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(state_vec)
                action = torch.argmax(q_values).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Punish if agent bumps into a wall (didn't change state)
        if next_state == state:
            reward -= 0.05  # light punishment for bumping

        # Encourage moving closer to goal
        def manhattan_distance(state_idx):
            row, col = divmod(state_idx, size)
            return abs(row - (size - 1)) + abs(col - (size - 1))

        reward += 5 / (manhattan_distance(next_state) + 1)

        # Heavily punish falling into a hole
        if terminated and reward == 0:
            reward = -5

        # Bonus for reaching goal
        if terminated and reward > 0:
            reward = 100
        next_state_vec = torch.tensor(one_hot(next_state, num_states), dtype=torch.float32)
        
        #Target Q-Value
        with torch.no_grad():
            target_q = reward + gamma * torch.max(q_net(next_state_vec))
            
        predicted_q = q_net(state_vec)[action]
        
        loss = loss_fn(predicted_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
torch.save(q_net.state_dict(), "frozenlake_qnet.pth")
print("Training complete")