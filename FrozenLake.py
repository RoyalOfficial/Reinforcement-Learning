import gymnasium as gym
import numpy as np
import time, os 

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode = "human")
q_table = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1 # learning rate
gamma = 0.99 # discoutn factor
epsilon = 0.1 # exploration factor
episodes = 1000

# Try to load saved Q-table
if os.path.exists("frozen_q_table.npy"):
    print("Loading saved Q-table...")
    q_table = np.load("frozen_q_table.npy")
else:
    print("No saved Q-table found. Training...")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    visited_states = set()
    
    while not done:
        current_state = state
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #explore
        else:
            action = np.argmax(q_table[state]) # expliot
    
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        #Punish Moving back and forth 
        if next_state == current_state:
            reward = -0.05
            
        #Punish Moving into Holes Heavily
        if terminated and reward == 0:
            reward = -1
            
        #Punish back tracking
        if next_state in visited_states:
            reward -= 0.1
           
        visited_states.add(next_state) 
        done = terminated or truncated
    
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
        state = next_state
        
    epsilon = max(0.01, epsilon * 0.995)
    np.save("frozen_q_table.npy", q_table)