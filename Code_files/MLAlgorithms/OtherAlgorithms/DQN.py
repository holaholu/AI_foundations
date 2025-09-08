#Description
#Deep Q-Network (DQN) is a type of  reinforcement learning algorithm that combines the power of deep learning/neutral networks with the principles of Q-learning. It uses a neural network to approximate the Q-values for each action in a given state allowing it to handle environments with high-dimensional and continuous state spaces. DQN uses experience (storing past experiences ans training on random batches) an a target network (to stabilize the learning process) to improve the performance of the algorithm. 

#Import necessary libraries
import gymnasium as gym  #provides a set of pre-built environments for reinforcement learning
import torch #provides a set of tools for deep learning
import torch.nn as nn #provides a set of tools for building neural networks
import torch.optim as optim #provides a set of tools for optimization
import numpy as np 
import random 

#Define the DQN Model
class DQN(nn.Module): # inherit from nn.Module making it a pytorch neural network model
    def __init__(self, input_size, output_size): # input_size is the number of features in the input, output_size is the number of features in the output
        super(DQN, self).__init__() # call the parent class constructor
        self.linear1 = nn.Linear(input_size, 64) # fully connected layer with 64 units output
        self.linear2 = nn.Linear(64, 32) # fully connected layer with 32 units output
        self.linear3 = nn.Linear(32, output_size) # fully connected layer with output_size units output
    
    def forward(self, x): # forward pass.forward pass is the process of passing the input through the network to get the output
        x = torch.relu(self.linear1(x)) # apply ReLU activation function to the output of the first linear layer. ReLU is a non-linear activation function that is used to introduce non-linearity into the model. It is defined as f(x) = max(0, x).
        x = torch.relu(self.linear2(x)) # apply ReLU activation function to the output of the second linear layer
        return self.linear3(x) # return the output of the third linear layer

#Hyperparameters
env_name = 'CartPole-v1' # environment name
learning_rate = 0.001 # learning rate is the step size used in the optimization process. It is a hyperparameter that controls the rate at which the model learns from the data. It is defined as the ratio of the change in the loss function to the change in the model parameters
gamma = 0.99 # discount factor is a hyperparameter that controls the trade-off between immediate rewards and future rewards. It is defined as the ratio of the future reward to the immediate reward.
buffer_size = 10000 # buffer size is the number of experiences stored in the replay buffer. It is a hyperparameter that controls the amount of experience stored in the replay buffer.
batch_size = 32 # batch size is the number of experiences used in the optimization process. It is a hyperparameter that controls the amount of experience used in the optimization process.
epsilon = 0.1 # epsilon is the exploration rate. It is a hyperparameter that controls the balance between exploration and exploitation. It is defined as the ratio of the exploration rate to the exploitation rate.
target_update_frequency = 100 # target update frequency is the number of experiences used in the optimization process. It is a hyperparameter that controls the amount of experience used in the optimization process.

#Initialize the environment and DQN
env = gym.make(env_name) # create the environment
input_size = env.observation_space.shape[0] # get the input size
output_size = env.action_space.n # get the output size based on number of possible actions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check if GPU is available.cuda means graphics processing unit.The name CUDA originally stood for "Compute Unified Device Architecture," but NVIDIA now uses it primarily as a brand name for its architecture and software tools that allow programmers to harness the massively parallel processing power of their GPUs for various compute-intensive tasks.
policy_net = DQN(input_size, output_size).to(device) # create the policy network. policy network is the network that is used to select actions based on the current state. It is created by connecting the input layer to the output layer.
target_net = DQN(input_size, output_size).to(device) # create the target network. target network is the network that is used to select actions based on the current state. 
target_net.load_state_dict(policy_net.state_dict()) # load the policy network weights into the target network
target_net.eval() # set the target network to evaluation mode. this disable gradient calculation and dropout. helps with stability.
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate) # create the optimizer. optimizer is the optimizer that is used to update the weights of the policy network. It is created by connecting the policy network to the optimizer.
criterion = nn.MSELoss() # create the loss function. loss function is the loss function that is used to measure the difference between the predicted and actual values. It is created by connecting the policy network to the loss function.

#Experience Replay Buffer
replay_buffer = [] # create the replay buffer. replay buffer is the buffer that stores the experiences. It is created by connecting the policy network to the replay buffer.

def train(num_episodes):
    step_count = 0 #initial step counter
    for episode in range(num_episodes):
        state, _ = env.reset(seed=42) # reset the environment with a fixed seed for reproducibility
        state = np.array(state) # convert the state to a numpy array
        done = False # flag to indicate if the episode is done
        total_reward = 0 # total reward for the episode
        while not done: # run until the episode is done
            #Epsilon-greedy action selection
            if random.random() < epsilon: # explore
                action = env.action_space.sample() # sample a random action
            else: # exploit
                with torch.no_grad(): # disable gradient calculation
                    action = policy_net(torch.tensor(state,  dtype=torch.float, device=device)).argmax().item() # select the action with the highest Q-value

            #Take action and get the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action) # take the action and get the next state and reward
            done = bool(terminated or truncated) # set the done flag as a Python boolean
            next_state = np.array(next_state) # convert the next state to a numpy array
            total_reward += reward # update the total reward

            #Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done)) # store the experience in the replay buffer
            if len(replay_buffer) > buffer_size: # if the replay buffer is full
                replay_buffer.pop(0) # remove the oldest experience

            #update current state
            state = next_state # update the current state

            #sample a batch of experiences from the replay buffer
            if len(replay_buffer) >= batch_size: # if the replay buffer is full
                batch = random.sample(replay_buffer, batch_size) # sample a batch of experiences from the replay buffer
                states, actions, rewards, next_states, dones = zip(*batch) # unzip the batch

                #convert to torch tensors and move to device
                states = torch.tensor(states, dtype=torch.float, device=device) # convert the states to a torch tensor and move to device
                actions = torch.tensor(actions, dtype=torch.long, device=device) # convert the actions to a torch tensor and move to device
                rewards = torch.tensor(rewards, dtype=torch.float, device=device) # convert the rewards to a torch tensor and move to device
                next_states = torch.tensor(next_states, dtype=torch.float, device=device) # convert the next states to a torch tensor and move to device
                dones = torch.tensor(dones, dtype=torch.float, device=device) # convert the dones to a torch tensor and move to device

                #Compute Q-values and target Q-values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)) # compute the Q-values for the current state and action
                next_q_values = target_net(next_states).max(1)[0].detach() # compute the Q-values for the next state and action
                target_q_values = rewards + gamma * next_q_values * (1 - dones) # compute the target Q-values

                #Compute loss and update policy network
                loss = criterion(current_q_values, target_q_values.unsqueeze(1)) # compute the loss
                optimizer.zero_grad() # zero the gradients
                loss.backward() # backpropagate the loss
                optimizer.step() # update the policy network

                #Update target network periodically
                step_count += 1 # increment the step counter
                if step_count % target_update_frequency == 0: # if the step counter is a multiple of the target update frequency
                    target_net.load_state_dict(policy_net.state_dict()) # update the target network
        print(f"Episode {episode}, Total Reward: {total_reward}") 


#Train the agent
train(num_episodes=1000)

                
                



                

            
                













        

        


        

    


