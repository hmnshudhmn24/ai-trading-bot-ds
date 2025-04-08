from agent import Agent
from env import TradingEnv
from utils import load_data

data = load_data('AAPL')
env = TradingEnv(data)
agent = Agent(env)
agent.train(episodes=50)