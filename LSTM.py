# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""
参考
https://mp.weixin.qq.com/s?__biz=MjM5MjAxMDM4MA==&mid=2651888917&idx=1&sn=cca0733e87f5bd4adacb213c22cf3536&chksm=bd48e4f68a3f6de01229eb015a50e18b294ce0de72454a2015a94ce5594c1c62d2f3fa8240e7&mpshare=1&scene=24&srcid=0920Afe7RQVF0ORbvo9Fy0uz&sharer_sharetime=1600583517911&sharer_shareid=824b452ab41934f064160ac1f64c08a8#rd
LSTM部分的code

作为进阶模型，我们将使用长短时记忆(LSTM)神经网络。在这里可以找到对LSTM的很好的介绍
(https：/machinelearningmaster ery.com/time-Series-prediction-lstm-rrurn-neuro-network-python-keras/)。
在这里，我们将窗口设置为6天，并让模型预测第7天。
"""""""""""""""""""""""""""


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




path='D:\\BaiduNetdiskDownload\\data.csv'
data=pd.read_csv(path)
data
 

 
"""""""""""""""""""""""""""""""""""""""""""""""""""
训练数据的构造部分

"""""""""""""""""""""""""""""""""""""""""""""""""""

#分割数据 80 20% 不随机混洗
n = len(data)
test_split=int(n*0.8)

#确定训练集和测试集
def get_window_data(data,window,edcs,preID):
    #确定窗口，归一化   
    #data=scaler.fit_transform(data.reshape(-1,1))
    n = len(data)
    x=[]
    y=[]
    for i in range(n-window-1):  #最后生成的列表的object数量1778-1-32
        a=data.iloc[i:i+window,edcs].values.tolist()
        b=data.iloc[i+window,preID]
        #import pdb; pdb.set_trace()
        #type(x)#debug
        x.append(a) 
        y.append(b)          
    return x,y

#归一化，min-max归一化
data_new= data.drop('date time',1) #数据出去日期非数值列准备归一化，
data_norm = (data_new - data_new.min()) / (data_new.max() - data_new.min())

#调用函数
window=32
edcs=[0,1,2,3,4,5,6]  #作为预测基础的edc的ID-1
#edcs=['1','2','3','4','5','6','7']  #作为预测基础的edc的ID
preID=0  # ID-1
x,y=get_window_data(data_norm,window,edcs,preID)
x=np.array(x) #这步总是失败cannot copy sequence with size 32 to array axis with dimension 7
y=np.array(y)
x_train=x[:test_split]
y_train=y[:test_split]
x_test=x[test_split:]
y_test=y[test_split:]

#变换数据reshape为LSTM的格式
x_train=np.reshape(x_train,(x_train.shape[0],window,7))




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
使用两层 LSTM 神经网络，激活函数选择 ReLU，
因为其计算量小且可以一定程度上避免梯度爆炸现象，
并且在两层 LSTM 之间加一层 dropout 层，并将失活率设置为 0.5
LSTM 的循环步数 num_step 设置为 32，即使用前 32 个负载数据预测接下来的负载，
LSTM 内部状态维度设置为128，即每个时间步的输入经过 LSTM 编码之后会被表示为一个 128 维的向量
LSTM输入格式[samples=train_X数据数量,timesteps=32,features=7]或(Batch_size, Time_step, Input_Sizes)
input_shape 参数 (free_batch_size, sequence_length, features_per_step)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def trainModel(x_train,y_train):
    """""""""""""""""""""
    trainX，trainY: 训练LSTM模型所需要的数据
    
    """""""""""""""""""""
    lstm_model=Sequential()
    lstm_model.add(LSTM(128,input_shape=(x_train.shape[1],x_train.shape[2]),\
                    activation='relu',\
                    return_sequences=True))
    lstm_model.add(Dropout(0.5))

    lstm_model.add(LSTM(128,activation='relu',return_sequences=False))

    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mse',optimizer='adam')

    #early_stop=EarlyStopping(monitor='loss',patience=4,verbose=1)

    history=lstm_model.fit(x_train,y_train,epochs=50,\
                                  batch_size=128,validation_split=0.004,\
                                  verbose=2,shuffle=False)
                                  #callbacks=[early_stop])
    return lstm_model,history





"""""""""""""""""""""""""""""""""""""""""""""""""""
训练model

"""""""""""""""""""""""""""""""""""""""""""""""""""

model,history=trainModel(x_train,y_train)




"""""""""""""""""""""""""""""""""""""""""""""""""""
plot history
选中多行后：　
Ctrl + 1: 注释/反注释
Ctrl + 4/5: 块注释/块反注释
Ctrl + L: 跳转到行号
Tab/Shift + Tab: 代码缩进/反缩进
"""""""""""""""""""""""""""""""""""""""""""""""""""
# plt.figure(figsize=(6,5),dpi=600)
# plt.plot(history.history['loss'],'darked',label='Train')
# plt.plot(history.history['val_loss'],'darkblue',label='Test')
# plt.title("Loss over epoch")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legent()
# plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""
make a prediction

"""""""""""""""""""""""""""""""""""""""""""""""""""

# yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)


#LSTM为7.26RMSE
