#Description
#Policy gradient methods are a class of reinforcement learning algorithms that use gradient-based optimization to find the optimal policy for a given environment. These methods are based on the idea that the policy can be represented as a probability distribution over actions, and that the optimal policy can be found by maximizing the expected return. Policy gradient methods are particularly useful for environments with high-dimensional and continuous state spaces, as they can handle these types of environments more effectively than other reinforcement learning algorithms. The REINFORCE algorithm is a popular policy gradient method that uses Monte Carlo sampling to estimate the gradient of the expected return with respect to the policy parameters.

#Import necessary libraries
import tensorflow as tf #provides a set of tools for deep learning
from tensorflow.keras import layers #provides a set of tools for building neural networks
import gymnasium as gym #provides a set of pre-built environments for reinforcement learning
import numpy as np #provides a set of tools for numerical computing
import random #provides a set of tools for random number generation

#Define the environment
env = gym.make('CartPole-v1') #create the environment
state_shape = env.observation_space.shape[0] #get the state shape
num_actions = env.action_space.n #get the number of actions

#Parameters
learning_rate = 0.01 #learning rate
gamma = 0.99 #discount factor


#Policy Network
def build_policy_model():
    model = tf.keras.Sequential([ #create the policy network where sequential means the layers are stacked on top of each other
        layers.Dense(24, activation='relu', input_shape=(state_shape,)), #first layer.24 is the number of nodes in the layer which means it can process 24 features at a time. activation='relu' means the activation function is ReLU (Rectified Linear Unit)
        layers.Dense(24, activation='relu'), #second layer
        layers.Dense(num_actions, activation='softmax') #output layer. softmax is used to convert the output of the network into a probability distribution over the possible actions
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model

policy_model = build_policy_model()

#function to select action based on policy
def choose_action(state):
    state = np.array(state).reshape(1, state_shape) #reshape the state to match the input shape of the policy network
    probabilities = policy_model.predict(state) #get the probabilities of each action
    return np.random.choice(num_actions, p=probabilities[0]) #select an action based on the probabilities

#function to calculate the return (discounted sum of rewards)
def discount_rewards(rewards):
    discounted = np.zeros_like(rewards) #initialize the discounted rewards array
    cumulative = 0 #initialize the running add
    for i in reversed(range(len(rewards))): #run through the rewards in reverse order
        cumulative = cumulative * gamma + rewards[i] #update the running add
        discounted[i] = cumulative #store the discounted reward
        return discounted - np.mean(discounted) #Normalize the discounted rewards

#Training function
def train_on_episode(state,actions,rewards):
    discounted_rewards = discount_rewards(rewards) #discount the rewards
    with tf.GradientTape() as tape: #record the operations for automatic differentiation
        action_probs = policy_model(tf.convert_to_tensor(state,dtype=tf.float32), training=True) #get the action probabilities for each state in the episode
        action_indices = tf.stack([tf.range(len(actions)), actions], axis=1) #get the indices of the actions
        selected_action_probs = tf.gather_nd(action_probs, action_indices) #get the probabilities of the selected actions
        loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * discounted_rewards) #calculate the loss
    gradients = tape.gradient(loss, policy_model.trainable_variables) #calculate the gradients of the loss with respect to the policy model's trainable variables
    
    policy_model.optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables)) #apply the gradients to the policy model

#Main training loop
num_episodes = 100 #number of episodes to train for
for episode in range(num_episodes): #run through the episodes
    state, _ = env.reset() #reset the environment
    episode_rewards,episode_actions,episode_states = [],[],[] #initialize the episode rewards, actions, and states arrays
    while True:
        action = choose_action(state) #select an action based on the policy
        next_state, reward, done, truncated, _ = env.step(action) #take the action and get the next state, reward, and done flag
        done = done or truncated #set the done flag as a Python boolean
        episode_rewards.append(reward) #append the reward to the episode rewards array
        episode_actions.append(action) #append the action to the episode actions array
        episode_states.append(state) #append the state to the episode states array
        state = next_state #update the state
        if done: #if the episode is done
            episode_states =np.vstack(episode_states) #stack the states vertically
            train_on_episode(episode_states,np.array(episode_actions),np.array(episode_rewards)) #train the policy model on the episode
            print(f"Episode {episode+1}, Total Reward: {sum(episode_rewards)}") #print the episode number and the total reward
            break #break out of the loop
    
    
    
        
