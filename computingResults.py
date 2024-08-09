import pandas as pd
import torch
# 导入需要的包
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.autograd  import Variable
from LSTMModel import LSTMModel
from parser_lstm import args
from dataset import getFabingShuData,getFabingBaiduData


def model_eval(model,test_loader,single_or_daidu,max_num, min_num):
    # 设置模型为评估模式
    model.eval()
    total_loss = 0
    preds = []  # 初始化预测值列表
    labels = []  # 初始化真实标签值列表
    # Evaluate data for one epoch
    for idx, (data, label) in enumerate(test_loader):
        if args.useGPU:
            # 单因素的数据需要升维
            if single_or_daidu == 1:
                data1 = data.unsqueeze(2)  # 删除data张量中的第一个维度，并将其移动到GPU
                pred = model(Variable(data1).cuda())  # 将data1封装成Variable并传入模型进行前向传播，得到预测值
            else:
                pred = model(Variable(data).cuda())  # 将data1封装成Variable并传入模型进行前向传播，得到预测值
            label = label.cuda()  # 将标签数据添加一个维度并移动到GPU
            # print(label.shape)
        else:
            pass
        criterion =nn.MSELoss(reduction='sum')
        loss = criterion(pred, label)
        total_loss += loss.item()
        preds.extend(pred.cpu().detach().numpy().tolist())
        labels.extend(label.cpu().detach().numpy().tolist())

    # # 计算评价指标
    preds = np.array(preds)
    preds = np.squeeze(preds)
    labels = np.array(labels)
    labels = np.squeeze(labels)
    # 如果是发病率，不需要下面两行代码
    preds = preds * (max_num - min_num) + min_num
    labels = labels * (max_num - min_num) + min_num

    rmse = np.sqrt(np.mean(np.square(preds - labels)))
    mae = np.mean(np.abs(preds - labels))
    mape = np.mean(np.abs(preds - labels) / labels) * 100
    '''
    rmse = torch.sqrt(nn.MSELoss()(torch.tensor(preds), torch.tensor(labels)))  # 均方根误差
    mae = torch.mean(torch.abs(torch.tensor(preds) - torch.tensor(labels)))  # 平均绝对误差
    mape = torch.abs((torch.tensor(labels) - torch.tensor(preds)) / torch.tensor(labels)).mean() * 100
    #mape = torch.abs((torch.tensor(labels) - torch.tensor(preds)) / torch.max(torch.tensor(labels),torch.tensor(0.000001))).mean() * 100
    '''
    # f.write('rmse: ' + str(rmse) + ' ')
    # f.write('mae: ' + str(mae) + ' ')
    # f.write('mape: ' + str(mape) + ' ')
    return rmse, mae, mape, preds

# 设置参数
args.datafile = r'./dataset/发病数原始数据.xlsx' # 发病率或是发病数的文件地址
file_path = "model/CV_ATT_LSTM_BD_num/"
save_path = ("./results/CV_ATT_LSTM_BD_num.xlsx")
single_or_daidu = 2   # 1 表示单因素，2表示双因素，即加百度指数

# 获取测试集的数据
if single_or_daidu==1:
    max_num,min_num,_, test_loader = getFabingShuData(args.datafile, args.sequence_length, args.batch_size)
elif single_or_daidu==2:
    max_num,min_num, _, test_loader = getFabingBaiduData(args.datafile, args.sequence_length, args.batch_size)
results = []
evaluations = []
for i in range(10):
    model = torch.load(file_path+"LSTM"+str(i)+".pth")
    rmse, mae, mape, preds = model_eval(model,test_loader,single_or_daidu,max_num, min_num)
    #rmse =rmse.tolist()
    #mae = mae.tolist()
    #mape = mape.tolist()
    #preds = [i.tolist() for i in preds]
    results.append(preds)
    evaluations.append([rmse, mae, mape])
# 求平均值
avgs = np.mean(np.array(evaluations), axis=0)
evaluations.append(avgs)
# 求方差
vars = np.var(np.array(evaluations), axis=0)
evaluations.append(vars)


# 将列表转换成dataframe,将结果保存到excel文件中
df = pd.DataFrame(evaluations)
df.to_excel(save_path,index=False,sheet_name='Sheet1',)

results = np.array(results)
results = np.squeeze(results)
df =pd.DataFrame(results)
with pd.ExcelWriter(save_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet2')


