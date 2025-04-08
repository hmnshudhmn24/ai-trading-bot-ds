# Autonomous AI-Powered Trading Bot

This project implements an autonomous AI-powered stock trading bot using reinforcement learning. The bot is designed to learn and make trading decisions (buy, sell, hold) based on historical stock market data. It uses Deep Q-Learning (DQN), a form of reinforcement learning, to optimize trading strategy over time.

## Features

- Uses reinforcement learning (DQN) for training a trading policy
- Fetches and preprocesses stock market data using `yfinance`
- Supports training, testing, and evaluation modes
- Simple interface for customization and experimentation

## Requirements

- Python 3.7+
- TensorFlow / PyTorch (TensorFlow used in this example)
- NumPy
- Pandas
- Matplotlib
- yfinance
- scikit-learn

## Setup

Install required packages using pip:

```bash
pip install -r requirements.txt
```

## How to Run

### Train the Model

```bash
python train.py
```

### Test the Model

```bash
python test.py
```

## Project Structure

- `train.py` - Training logic for the trading agent
- `test.py` - Testing logic for the agent on unseen data
- `model.py` - DQN model definition
- `agent.py` - Agent logic and reinforcement learning implementation
- `env.py` - Trading environment
- `utils.py` - Helper functions for preprocessing
- `requirements.txt` - Python dependencies

## Disclaimer

This project is for educational purposes only and should not be used for actual trading without significant modification and thorough testing.