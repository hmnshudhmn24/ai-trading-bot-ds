import numpy as np

class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.balance = 1000
        self.shares_held = 0
        self.initial_balance = 1000
        self.observation_space = data.shape[1]
        self.action_space = 3  # Buy, Sell, Hold

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        return self.data[self.current_step]

    def step(self, action):
        current_price = self.data[self.current_step][3]  # Use Close price
        reward = 0

        if action == 0:  # Buy
            if self.balance > current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 1:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                reward = 1
        # Hold: do nothing

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        next_state = self.data[self.current_step]
        return next_state, reward, done