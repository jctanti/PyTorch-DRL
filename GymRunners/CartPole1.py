# Simple CartPole environment with 1-step Q learning
# Justin Tantiongloc

import gym
import time
import sys
import numpy as np
from collections import deque
sys.path.insert(0, './Agents/')
from QLearn1 import QLearn1
from ExperienceReplay import ExperienceReplay
import random

# Params
MAX_EPS = 10000
MAX_STEPS = 1000
MAX_SIZE = 10000
EXP_DRAW = 32

env = gym.make('CartPole-v0')
env.seed(int(time.time()))

# Get dims
action_dim = env.action_space.n
space_dim = len(env.reset())
model = QLearn1(space_dim, action_dim, [32, 16])
expRep = ExperienceReplay(MAX_SIZE)
model.printParams()

for cur_ep in range(MAX_EPS):
    cur_obs = env.reset()
    cur_ep_reward = 0

    for step in range(MAX_STEPS):
        env.render()

        # Get action from state
        new_action = model.getAction(cur_obs)

        #Run new action through env
        new_obs, rew, done, info = env.step(new_action)

        #Add to expRep
        expRep.addExperience(cur_obs, new_action, new_obs, rew, done)

        # Accumulate reward
        cur_ep_reward += rew

        if expRep.getSize() > EXP_DRAW:
            train_data = expRep.drawExperiences(EXP_DRAW)
            model.learnStep(train_data)

        # If terminate or end of ep, print/record reward of this ep
        if done or step == MAX_STEPS - 1:
            print("Episode %d reward: %d, epsilon: %f" % (cur_ep, cur_ep_reward, model.epsilon))
            break
        cur_obs = new_obs

env.close()
