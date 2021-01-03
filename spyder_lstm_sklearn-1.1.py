# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:29:08 2020

@author: qjncn
"""


import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
tf.random.set_seed(2345)

path='D:\\BaiduNetdiskDownload\\data_new.csv'
data_new=pd.read_csv(path)

"""""""""""""""""""""""""""""""""""""""""""""""""""
训练数据的构造部分

"""""""""""""""""""""""""""""""""""""""""""""""""""

#处理数据归一化，确定训练集和测试集
def get_window_data(data,window,edcs,preID,test_split):
    """""""""""""""""""""""
    input:  data   数据
            window 窗口长度
            edcs   作为神经网络模型输入的最为预测基础的edc的ID-1
            preID  作为神经网络模型输出的需要预测的edc的ID-1  
            test_split 分割系数
    return: x_train,y_train,x_test,y_test
            
    """""""""""""""""""""""
    #归一化，min-max归一化 后在0-1之间
    #data_new = data.drop('date time',1) #数据除去日期非数值列准备归一化，
    ###data_norm = (data_new - data_new.min()) / (data_new.max() - data_new.min())# 数值在0-1之间
    #data=scaler.fit_transform(data_new.reshape(0,1))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)#type(scaled_data) =<class 'numpy.ndarray'>
    #构造数据x和标签y
    n = len(scaled_data)
    x=[]
    y=[]
    for i in range(n-window-1):  #最后生成的列表的object数量1778-1-32
        #a=scaled_data.iloc[i:i+window,edcs].values.tolist()
        a=scaled_data[i:i+window,edcs]
        b=scaled_data[i+window,preID]
        #import pdb; pdb.set_trace()
        #type(x)#debug
        x.append(a) 
        y.append(b) 
        
    x=np.array(x) 
    y=np.array(y)
       
    
    #分割数据
    x_train=x[:test_split]
    y_train=y[:test_split]
    x_test=x[test_split:]
    y_test=y[test_split:]
    return x_train,y_train,x_test,y_test



#调用函数
window=32
edcs=[0,1,2,3,4,5,6]  #作为预测基础的edc的ID-1
#edcs=['1','2','3','4','5','6','7']  #作为预测基础的edc的ID
preID=0  # ID-1
n = len(data_new)
test_split=int(n*0.8)
x_train,y_train,x_test,y_test=get_window_data(data_new,window,edcs,preID,test_split)

#变换数据reshape为LSTM的格式
x_train=np.reshape(x_train,(x_train.shape[0],window,7))   #  x_train.shape  =  (1422, 32, 7)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
使用两层 LSTM 神经网络，激活函数选择 ReLU，
因为其计算量小且可以一定程度上避免梯度爆炸现象，
并且在两层 LSTM 之间加一层 dropout 层，并将失活率设置为 0.5
LSTM 的循环步数 num_step 设置为 32，即使用前 32 个负载数据预测接下来的负载，
LSTM 内部状态维度设置为128，即每个时间步的输入经过 LSTM 编码之后会被表示为一个 128 维的向量
LSTM输入格式[samples=train_X数据数量,timesteps=32,features=7]或(Batch_size, Time_step, Input_Sizes)
input_shape 参数 (free_batch_size, sequence_length, features_per_step)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
def create_model(neurons=256,lstm_num=2):
    """""""""""""""""""""
    input: trainX，trainY: 训练LSTM模型所需要的数据
    return: lstm_model, history
    
    """""""""""""""""""""
    model=Sequential()
    model.add(LSTM(neurons=neurons,input_shape=(x_train.shape[1],x_train.shape[2]),\
                                        activation='linear',\
                                        return_sequences=True))
   # model.add(Dropout(dropout_rate))

    model.add(LSTM(neurons=neurons,activation='linear',return_sequences=False))
    
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='Adam')

    #early_stop=EarlyStopping(monitor='loss',patience=4,verbose=1)

    #history=lstm_model.fit(x_train,y_train,epochs=125,\
                                 # batch_size=16,validation_split=0.2,\
                                 # verbose=2,shuffle=True)
                                  #callbacks=[early_stop])
    return model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
网格搜索参数优化
https://blog.csdn.net/weixin_38664232/article/details/87868355
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# fix random seed for reproducibilit
seed = 7
numpy.random.seed(seed)
# create model
model = KerasRegressor(build_fn=create_model, verbose=2)
# define the grid search parameters
neurons = [16, 32, 64, 128, 256]
#weight_constraint = [1, 2, 3, 4, 5]
#优化网络权值初始化
#优化学习速率和动量因子
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#activation = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#batch_size = [10, 20, 40, 60, 80, 100]
epochs = [150,200,250]
lstm_num=[2,3,4]
param_grid = dict(batch_size=batch_size,nb_epoch=epochs,optimizer=optimizer,neurons=neurons,activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
