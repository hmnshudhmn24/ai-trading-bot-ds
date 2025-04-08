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
