from collections import deque
import numpy as np
class ExperienceReplay:
    def __init__(self, max_size = 10000):
        self.data = deque(maxlen=max_size)
    
    def addExperience(self, start_state, action, end_state, reward, done):
        self.data.appendleft([start_state, action, end_state, reward, done])
    
    def getSize(self):
        return len(self.data)

    def drawExperiences(self, num_draw):
        indices = np.random.randint(0, len(self.data), num_draw)
        output={}
        output['new_states'] = []
        output['old_states'] = []
        output['rewards'] = []
        output['terminals'] = []
        output['actions'] = []
        for i in range(num_draw):
            output['new_states'].append(self.data[indices[i]][2])
            output['old_states'].append(self.data[indices[i]][0])
            output['rewards'].append(self.data[indices[i]][3])
            output['terminals'].append(self.data[indices[i]][4])
            output['actions'].append(self.data[indices[i]][1])
        output['new_states'] = np.array(output['new_states'])
        output['old_states'] = np.array(output['old_states'])
        output['rewards'] = np.array(output['rewards'])
        output['terminals'] = np.array(output['terminals'])
        output['actions'] = np.array(output['actions'])

        return output