import pandas as pd
import numpy as np
import pylab, matplotlib.pyplot
import matplotlib as plt
import datetime
import sys
import torch
#sys.path.append("../FinRL-Library")
matplotlib.use('Agg')
%matplotlib inline
from finrl.config import config
from finrl.marketdata.tusharedownloader import TushareDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrade import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import argparse
import itertools

import os

 #create folder 引入目录结构

if not os.path.exists("./" + config.DATA_SAVE_DIR):
 os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
 os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
 os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
 os.makedirs("./" + config.RESULTS_DIR)


 #download data

 df = TushareDownloader(start_date = '2010-01-01',
 end_date = '2021-07-11',
 ticker_list = config.SSE_20_TICKER).fetch_data()
df.shape
df.sort_values(['date','tic'],ignore_index=True).head()

# 引入技术指标：MACD和RSI，添加湍流指数。
fe = FeatureEngineer(
 use_technical_indicator=True,
 tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
 use_turbulence=True,
 user_defined_feature = False)
​
processed = fe.preprocess_data(df)
​
#导入环境

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))
​
processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])
​
processed_full = processed_full.fillna(0)

#split.data

train = data_split(processed_full, '2009-01-01','2021-04-01')
trade = data_split(processed_full, '2021-04-01','2021-07-11')
print(len(train))
print(len(trade))

# 计算出股票的空间，后续灌入模型进行强化学习
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

#set env

env_kwargs = {
 "hmax": 100, 
 "initial_amount": 1000000, 
 "buy_cost_pct": 0.001,
 "sell_cost_pct": 0.001,
 "state_space": state_space, 
 "stock_dim": stock_dimension, 
 "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
 "action_space": stock_dimension, 
 "reward_scaling": 1e-4
 
}
​
e_train_gym = StockTradingEnv(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# set agent,选用TD3算法进行训练

agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
agent = DRLAgent(env = env_train)
model_td3 = agent.get_model("td3")

#训练

trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=50000)
trained_ppo.save(f"{config.TRAINED_MODEL_DIR}/ppo")

#trade
trade = data_split(processed_full, '2020-05-01','2021-07-11')
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 380, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

#action
df_actions.head()
df_actions.to_csv("./datasets/actions"+'.csv')

df_account_value.tail()

# account value

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
​
perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

"""Annual return 年回报率

Cumulative returns累计回报

Annual volatility年波动率

Sharpe ratio夏普比率

Calmar ratio卡尔马尔比率

Stability稳定性

Max drawdown 最大压降"""

## plot

def plot(tradedata,actionsdata,ticker):    
 #the first plot is the actual close price with long/short positions
 # 绘制实际的股票收盘数据
 df_plot = pd.merge(left=tradedata ,right=actionsdata,on='date',how='inner')
 plot_df = df_plot.loc[df_plot['tic']==ticker].loc[:,['date','tic','close',ticker]].reset_index()
 fig=plt.figure(figsize=(12, 6))
 ax=fig.add_subplot(111)    
 ax.plot(plot_df.index, plot_df['close'], label=ticker)
 # 只显示时刻点，不显示折线图 => 设置 linewidth=0
 ax.plot(plot_df.loc[plot_df[ticker]>0].index, plot_df['close'][plot_df[ticker]>0], label='Buy', linewidth=0, marker='^', c='g')
 ax.plot(plot_df.loc[plot_df[ticker]<0].index, plot_df['close'][plot_df[ticker]<0], label='Sell', linewidth=0, marker='v', c='r')
 
 plt.legend(loc='best')
 plt.grid(True)
 plt.title(ticker +'__'+plot_df['date'].min()+'___'+plot_df['date'].max())
 plt.show()
 print(plot_df.loc[df_plot[ticker]>0])
