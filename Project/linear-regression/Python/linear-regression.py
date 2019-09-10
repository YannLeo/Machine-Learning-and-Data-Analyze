#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# function:
# 读取数据，并对数据进行线性回归
#

import numpy as np
import os
import pandas as pd
import math


# @brief: 
# 获取数据文件路径
#
def get_data_path():
    path = os.path.abspath('..')
    path = path + '\\Dataset'
    path = path + '\\' + os.listdir(path)[0]
    return path

# @brief: 
# 读取数据至workspace
#
def read_data(path):                    
    data = pd.read_csv(path)
    return data.values

# @brief: 
# 训练模型参数，使得MSE最低，
# 使用批量梯度下降法进行参数训练
#
def trainning(train, LR):               # 默认最后一列是因变量，前几列是自变量
    print('Tranning:   [Loading Model...]')
    parameter = np.random.rand(train.shape[1])   # 线性模型参数
    x = np.hstack((np.ones([train.shape[0],1]),train[:,0:-1]))  # 取出自变量，并加上常数项权重1
    y = train[:,-1]                                             # 取出因变量
    y_hat = np.dot(x,parameter)
    MSE = caculate_MSE(y, y_hat)        # 计算MSE，判断模型是否已训练好
    times = 0
    print('\n[Trainning Model...]')
    while MSE > 2 and times <= 500:
        if times % 5 == 0:
            print('Times:', times, ',\tMSE:', MSE)
        step = get_descent_step(x, y, y_hat)        # 获取梯度下降的反方向和步长
        parameter = parameter - LR * step           # 更新权重
        y_hat = np.dot(x,parameter)                 
        MSE = caculate_MSE(y, y_hat)
        times += 1
    return parameter

# @brief: 
# MSE对各个参数进行求导，确定下降方向和步长
#
def get_descent_step(x, y, y_hat):
    NUM = x.shape[0]
    x_real = x[:,1:]        # x_real为去掉常数项后的自变量集合
    da0 = 2 * np.sum(y_hat - y) / NUM   # 常数项的导数为 2/N*sigma(i=1 to N){y_hat_i-y_i}
    step = np.hstack((da0, 2*np.dot(y_hat-y,x_real)/NUM))   # 非常数项导数为 2/N*sigma(i=1 to N){(y_hat_i-y_i)*x_i}
    return step

# @brief: 
# 计算MSE
#
def caculate_MSE(y, y_hat):
    NUM = y_hat.shape
    minus = y - y_hat
    square = np.square(minus)
    MSE = np.sum(square) / NUM
    return MSE

# @brief: 
# 对数据进行归一化
# 归一化采用0,1归一化，即(x-minnum)/(maxnum-minnum)
#
def normalize(train):
    x = train[:,0:-1]
    maxnum = x.max(axis=0)
    minnum = x.min(axis=0)
    parameter_normalize = np.vstack((minnum,maxnum-minnum))
    train = np.hstack(((x - minnum) / (maxnum - minnum), train[:,-1][:,np.newaxis]))
    return train, parameter_normalize

# @brief: 
# 对学习模型进行测试，观察模型扩展性
#
def testing(parameters, parameter_normalize, test):
    print('\n[Testing Model...]')
    test_x = test[:,0:-1]
    test_y = test[:,-1]
    test_x = (test_x - parameter_normalize[0,:]) / parameter_normalize[1,:]
    test_x = np.hstack((np.ones((test_x.shape[0],1)),test_x))
    test_y_hat = np.dot(test_x, parameters)
    test_MSE = caculate_MSE(test_y, test_y_hat)
    NUM = test_y_hat.shape
    minus = np.abs(test_y - test_y_hat)
    ERROR = np.sum(minus) / NUM
    ERROR_PERCNET = ERROR / np.sum(test_y)
    print("\nThe test reports of the linear model are as follows:")
    print("The MSE of the test dataset is\t\t",test_MSE)
    print("The error rate of test dataset is \t",ERROR_PERCNET,"\n")

# @brief: 
# 显示模型参数
#
def show_model(parameters, data):
    col = []
    for i in range(data.shape[1]):
        col.append('a'+str(i))
    df_parameters = pd.DataFrame(parameters.reshape(1,data.shape[1]),columns=col)
    print("\nThe parameters of the linear regression model are as follows")
    print(df_parameters)

def save_model():
    pass

def load_model():
    pass

# @brief: 
# 获取数据文件路径
#
if __name__ == "__main__":
    path  = get_data_path()
    data  = read_data(path)      # data的类型是numpy.ndarray
    train = data[0:math.floor(data.shape[0]*0.8),:]
    train, parameter_normalize = normalize(train)
    test  = data[math.floor(data.shape[0]*0.8):,:]
    Learning_Rate = 0.1
    parameters = trainning(train, Learning_Rate)
    show_model(parameters, data)
    testing(parameters, parameter_normalize, test)
