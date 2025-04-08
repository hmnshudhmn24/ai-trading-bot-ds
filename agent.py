import numpy as np
from model import create_model
import random
from collections import deque

class Agent:
    def __init__(self, env):
        self.env = env
        self.model = create_model(env.observation_space, env.action_space)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.action_space)
        q_values = self.model.predict(state[np.newaxis])
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis])[0])
            target_f = self.model.predict(state[np.newaxis])
            target_f[0][action] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            self.replay()
            print(f"Episode {e+1}/{episodes}, Reward: {total_reward}")

    def test(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.act(state)
            next_state, reward, done = self.env.step(action)
            state = next_state
            total_reward += reward
        print("Test reward:", total_reward)