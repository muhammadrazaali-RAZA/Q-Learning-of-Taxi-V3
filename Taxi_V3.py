#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imort Required Packages
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output


# In[2]:


env  = gym.make('Taxi-v3')


# In[3]:


episodes =10

for episode in range(6,episodes):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        score += reward
        clear_output(wait=False)
    print('Episode: {}\nScore: {}'.format(episode,score))
    
env.close()
        


# In[4]:


#Create Q-Table

actions = env.action_space.n
state = env.observation_space.n


q_table = np.zeros((state,actions))


# In[5]:


#parameters for Q-Learning

num_episodes = 1000000
max_steps_per_episode = 1000

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []


# In[6]:


#Q-Learning Algo

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode):
        
        #Exploration Vs Exploitation trade-off
        exploration_threshold =random.uniform(0,1)
        if exploration_threshold > exploration_rate :
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        #update Q-Table
        q_table[state,action]  = q_table[state,action] * (1-learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
        
        state = new_state
        
        rewards_current_episode += reward
        
        if done ==True:
            break
    
    exploration_rate = min_exploration_rate +                         (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    rewards_all_episodes.append(rewards_current_episode)
    
print("*************  Traning Finished **************")


# In[14]:


q_table.shape


# In[8]:


#calculate and average reward per thousand episodes

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("Average per thousand episodes")
for r in rewards_per_thousand_episodes:
    print(count, " : ",str(sum(r/1000)))
    count+=1000


# In[13]:


#Visualize Agent

import time

for episode in range(2):
    state =env.reset()
    done = False
    print("Episode is: " + str(episode))
    time.sleep(1)
    
    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.7)
        
        action = np.argmax(q_table[state,:])
        
        new_state, reward, done, info =env.step(action)
        
        if done:
            clear_output(wait=True)
            env.render()
            if reward > 0:
                print("Reward: ", reward)
                print("******** Reached Goal *******")
                time.sleep(2)
                clear_output(wait=True)
            else:
                print("Reward: ", reward)
                print("********* Faild  ***********")
                time.sleep(2)
                clear_output(wait=True)
            
            break
            
        state =new_state
env.close()


# In[15]:


# Save Q-Table of Taxi 
np.savetxt('Q_table_taxi.txt', q_table)


# In[16]:


#load Q-Table of Taxi
markers = np.fromfile("Q_table_taxi.txt")


# In[ ]:




