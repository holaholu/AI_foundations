#Description
#Q-learning is a type of model-free reinforcement learning algorithm that can be used to learn the optimal policy for a given environment. It works by iteratively updating the Q-value of each state-action pair based on the rewards received and the maximum Q-value of the next state. It updates the Q-table. Q-learning is useful for learning the optimal policy for a given environment.It is effective where the environment is Markovian (the next state depends only on the current state and action) and can be represented by discrete states and actions.

#Import necessary libraries
import numpy as np
import random

#Define the environment
num_states = 16 #4x4 grid
num_actions = 4 #UP,RIGHT,DOWN,LEFT
q_table = np.zeros((num_states, num_actions))

#Define the parameters
alpha = 0.1 #learning rate. Controls the balance between the new Q-value and the old Q-value
epsilon = 0.1 #exploration rate. Controls the balance between a random action and the best action known so far
gamma = 0.9 #discount factor. controls how much weight future rewards have versus immediate rewards
num_episodes = 1000 #number of episodes/iterations to run the algorithm

#Define a simple reward structure
rewards = np.zeros(num_states) #initialize reward array with zeros
rewards[15] = 1 #set the reward for the goal state to 1

#Function to determine the next state based on the current state and action
def get_next_state(state, action):
    if action == 0 and state>=4: #up. state>=4 ensures we don't go out of bounds
        return state-4 #move up by subtracting 4 from the current state
    elif action == 1 and (state + 1) % 4 !=0: #Right. (state + 1) % 4 !=0 ensures we don't go out of bounds
        return state+1 #move right by adding 1 to the current state
    elif action == 2 and state < 12: #Down state<12 ensures we don't go out of bounds
        return state+4 #move down by adding 4 to the current state
    elif action == 3 and state % 4 != 0: #Left. state % 4 != 0 ensures we don't go out of bounds
        return state-1 #move left by subtracting 1 from the current state
    else:
        return state    #Stay in the same state if action is not possible/out of bounds
    
#Q-learning algorithm
for episode in range(num_episodes):
    state = random.randint(0, num_states-1) #randomly select a state
    while state != 15: #run until the goal state is reached
        if random.uniform(0, 1) < epsilon: #Explore
            action = random.randint(0, num_actions-1) #randomly select an action
        else: #Exploit
            action = np.argmax(q_table[state]) #select the action with the highest Q-value

        next_state = get_next_state(state, action) #get the next state based on the current state and action
        reward = rewards[next_state] #get the reward for the next state
        old_value = q_table[state][action] #get the old Q-value
        next_max = np.max(q_table[next_state]) #get the maximum Q-value of the next state

        #Q-learning Update rule
        new_value = old_value + alpha * (reward + gamma * next_max - old_value) #update the Q-value
        q_table[state][action] = new_value #update the Q-table  
        state = next_state #move to the next state

#Display the learned Q-table
print("Learned Q-table:")
print(q_table)

            
            









