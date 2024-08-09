from pandas import read_excel
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np
from torchvision import transforms
from parser_lstm import args

# 获取发病数的数据集，返回的是最大值、最小值和train_loader、test_loader
def getFabingShuData(dataFile, sequence_length, batchSize):   # 单因素预测数据集制作，发病数和发病率都用一个函数
    #fabingshu_data = read_excel('./dataset/发病率原始数据.xlsx')
    fabingshu_data = read_excel(dataFile)
    #fabingshu_data = fabingshu_data['发病率'] # 发病率或是发病数
    fabingshu_data = fabingshu_data.iloc[:,1]
    fabingshu_max = fabingshu_data.max()  # 收盘价的最大值
    fabingshu_min = fabingshu_data.min()  # 收盘价的最小值
    fabingshu_data = (fabingshu_data - min(fabingshu_data)) / (max(fabingshu_data) - min(fabingshu_data))

    #sequence = 4  # 天数
    sequence = sequence_length
    X = []  # 特征
    Y = []  # 标签

    # 遍历数据，创建基于前四个月的数据集
    for i in range(sequence, len(fabingshu_data)):
        temp_x = fabingshu_data[i - sequence:i]
        temp_y = fabingshu_data[i]
        #print(temp_x)
        #print(temp_y)
        X.append(temp_x)
        Y.append(temp_y)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    # 发病率的时候直接使用原始值，发病数的时候，使用标准化的值
    #Y = Y*(fabingshu_max - fabingshu_min)+fabingshu_min   # 原始值

    # 构建训练集和测试集
    total_len = X.shape[0]  # 总数据量

    # print(total_len)
    trainx, trainy = X[:total_len - 30], Y[:total_len - 30]  # 训练集
    index = list(range(len(trainx)))
    random.shuffle(index)

    trainx, trainy = trainx[index[:int(len(index) * 0.90)]], trainy[index[:int(len(index) * 0.90)]]

    testx, testy = X[-30:], Y[-30:]  # 测试集
    #print(trainx.shape, trainy.shape, testx.shape, testy.shape)

    train_loader = DataLoader(dataset=Mydataset(trainx, trainy), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=False)  # 测试集数据加载器
    return fabingshu_max, fabingshu_min, train_loader, test_loader  # 返回最大值、最小值、训练集和测试集数据加载器


# 获取发病数的数据集，返回的是最大值、最小值和train_loader、test_loader
def getFabingBaiduData(dataFile, sequence_length, batchSize):
    # fabingshu_data = read_excel('./dataset/发病率原始数据.xlsx')
    fabingshu_data = read_excel(dataFile)
    #fabingshu_data = fabingshu_data[['发病率','处理后百度指数']]
    fabingshu_data = fabingshu_data.iloc[:,1:3]

    fabingshu_max = fabingshu_data.iloc[:,0].max()  # 发病最大值
    fabingshu_min = fabingshu_data.iloc[:,0].min()  # 发病最小值
    fabingshu_data = fabingshu_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    #sequence = 4  # 天数
    sequence = sequence_length
    X = []  # 特征
    Y = []  # 标签

    # 遍历数据，创建基于前四个月的数据集
    for i in range(sequence, len(fabingshu_data)):
        temp_x = fabingshu_data.iloc[i - sequence:i, ]
        temp_y = fabingshu_data.iloc[i][0]
        #print(temp_x)
        #print(temp_y)
        X.append(temp_x)
        Y.append(temp_y)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # 发病率的时候使用该语句，发病数的时候不适用
    #Y = Y*(fabingshu_max - fabingshu_min)+fabingshu_min

    # 构建训练集和测试集
    total_len = X.shape[0]  # 总数据量

    # print(total_len)
    trainx, trainy = X[:total_len - 30], Y[:total_len - 30]  # 训练集
    index = list(range(len(trainx)))
    random.shuffle(index)

    trainx, trainy = trainx[index[:int(len(index) * 0.90)]], trainy[index[:int(len(index) * 0.90)]]

    testx, testy = X[-30:], Y[-30:]  # 测试集
    #print(trainx.shape, trainy.shape, testx.shape, testy.shape)

    train_loader = DataLoader(dataset=Mydataset(trainx, trainy), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=False)  # 测试集数据加载器
    return fabingshu_max, fabingshu_min, train_loader, test_loader  # 返回最大值、最小值、训练集和测试集数据加载器

class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = torch.tensor(xx, dtype=torch.float32)  # 特征数据
        self.y = torch.tensor(yy, dtype=torch.float32).unsqueeze(1)  # 标签数据

    def __getitem__(self, index):
        x1 = self.x[index]  # 获取指定索引的特征数据
        y1 = self.y[index]  # 获取指定索引的标签数据
        return x1, y1  # 否则，直接返回数据

    def __len__(self):
        return len(self.x)  # 返回数据集的长度


class LSTMDataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = torch.tensor(xx, dtype=torch.float32)
        self.y = torch.tensor(yy, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

