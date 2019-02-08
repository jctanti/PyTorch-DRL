import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# DRL model: FC -> relu -> FC -> relu -> FC -> output
class QModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#RL Agent
class QLearn1:
    def __init__(self, state_dim, action_dim, layers, gamma=.9, learning_rate = 1e-3, epsilon = 1.0, epsilon_decay = .995, epsilon_min = .1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = layers
        self.learning_rate = learning_rate

        self.buildModel()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    #prints params (really just a test to see params)
    def printParams(self):
        for name, params in self.model.named_parameters():
            print(name, params.data)

    # Create deep learning model and set up loss + optimization stuff
    def buildModel(self):
        self.model = QModel(self.state_dim, self.action_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # Takes a sample batch and learns 1-step Q learning from it
    def learnStep(self, train_data):
        self.optimizer.zero_grad() # don't forget this line, or everything blows up

        #eval target states and starting states for computation of loss
        q_new = self.model(torch.from_numpy(train_data['new_states']).float().view(-1,self.state_dim))
        q_old = self.model(torch.from_numpy(train_data['old_states']).float().view(-1,self.state_dim))
        q_target = q_old.clone()

        #evaluate based on terminal states, and with 1-step return
        for i in range(len(train_data['old_states'])):
            temp = train_data['rewards'][i]
            if not train_data['terminals'][i]:
                temp = train_data['rewards'][i] + self.gamma * q_new[i].max(0)[0].item()
            q_target[i][train_data['actions'][i]] = temp

        #get loss, populate grads, and take optimization step
        loss = self.criterion(q_old, q_target)
        loss.backward()
        self.optimizer.step()

        #adjust exploration
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
    
    # argmax over Q function with state input (or random)
    def getAction(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            return self.model(torch.from_numpy(state).float().view(1,-1))[0].max(0)[1].item()

