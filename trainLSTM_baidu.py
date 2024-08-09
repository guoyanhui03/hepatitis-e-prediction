# 导入需要的包
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd  import Variable
from LSTMModel import LSTMModel
from parser_lstm import args
from dataset import getFabingShuData,getFabingBaiduData
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/lstm')

def model_eval(model,test_loader,max_num, min_num):
    # 设置模型为评估模式
    model.eval()
    total_loss = 0
    preds = []  # 初始化预测值列表
    labels = []  # 初始化真实标签值列表
    # Evaluate data for one epoch
    for idx, (data, label) in enumerate(test_loader):
        if args.useGPU:
            # 单因素的数据需要升维
            #data1 = data.unsqueeze(2)  # 删除data张量中的第一个维度，并将其移动到GPU
            #pred = model(Variable(data1).cuda())  # 将data1封装成Variable并传入模型进行前向传播，得到预测值
            # 加百度指数的数据不需要升维
            pred = model(Variable(data).cuda())  # 将data1封装成Variable并传入模型进行前向传播，得到预测值
            # print(pred.shape)
            label = label.cuda()  # 将标签数据添加一个维度并移动到GPU
            # print(label.shape)
        else:
            pass
        criterion =nn.MSELoss(reduction='sum')
        loss = criterion(pred, label)
        total_loss += loss.item()
        # 将GPU的tensor转化成列表
        preds.extend(pred.cpu().detach().numpy().tolist())
        labels.extend(label.cpu().detach().numpy().tolist())
    avg_loss = total_loss / len(test_loader.dataset)
    # writer.add_scalar('Loss/eval', loss.item())
    # f.write('Loss/eval: ' + str(avg_loss) + ' ')
    # # 计算评价指标
    preds = np.array(preds)
    preds = np.squeeze(preds)
    labels = np.array(labels)
    labels = np.squeeze(labels)

    preds = preds * (max_num - min_num) + min_num
    labels = labels * (max_num - min_num) + min_num

    rmse = np.sqrt(np.mean(np.square(preds - labels)))
    mae = np.mean(np.abs(preds - labels))
    mape = np.mean(np.abs(preds - labels) /labels)*100
    '''
    rmse = torch.sqrt(nn.MSELoss()(torch.tensor(preds), torch.tensor(labels)))  # 均方根误差
    mae = torch.mean(torch.abs(torch.tensor(preds) - torch.tensor(labels)))  # 平均绝对误差
    mape = torch.abs((torch.tensor(labels) - torch.tensor(preds)) / torch.tensor(labels)).mean() * 100
    #mape = torch.abs((torch.tensor(labels) - torch.tensor(preds)) / torch.max(torch.tensor(labels),torch.tensor(0.000001))).mean() * 100
    '''
    # f.write('rmse: ' + str(rmse) + ' ')
    # f.write('mae: ' + str(mae) + ' ')
    # f.write('mape: ' + str(mape) + ' ')
    return rmse, mae, mape, avg_loss

#def train():
def train(args):
    min_mape = 1000
    # 创建模型
    model = LSTMModel(args.input_size, args.hidden_size, args.num_layers,args.output_size)
    model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 获取训练集和测试机的数据加载器
    max_num, min_num, train_loader, test_loader = getFabingBaiduData(args.dataFile, args.sequence_length,
                                                              args.batch_size)

    # 初始化列表用于记录每个epoch的总损失
    epoch_losses = []

    with open('LSTM_train_loss_log.txt', 'a') as f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            for idx, (data, label) in enumerate(train_loader):
                if args.useGPU:
                    #data1 = data.squeeze(1).cuda()  # 删除data张量中的第一个维度，并将其移动到GPU
                    # 单因素数据需要升维
                    #data1 = data.unsqueeze(2)
                    #pred = model(Variable(data1).cuda())  # 将data1封装成Variable并传入模型进行前向传播，得到预测值

                    # 加百度指数的数据不需要升维
                    pred = model(Variable(data).cuda())  # 将data1封装成Variable并传入模型进行前向传播，得到预测值
                    #print(pred.shape)
                    label = label.cuda()  # 将标签数据添加一个维度并移动到GPU
                    #print(label.shape)
                else:  # 如果使用CPU
                    pass
                loss = criterion(pred, label)  # 计算当前batch的损失值
                optimizer.zero_grad()  # 清空优化器的梯度
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新模型参数
                total_loss += loss.item()  # 累加当前batch的损失值到total_loss
            avg_train_loss = total_loss / len(train_loader.dataset)

            # 记录每个epoch的总损失
            #epoch_losses.append(avg_loss)
            # 记录每个epoch的评估指标和测试集的损失
            rmse, mae, mape, avg_eval_loss = model_eval(model,test_loader,max_num, min_num)

            # tensorboard记录损失变化
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/eval', avg_eval_loss, epoch)

            # 将损失写入文件
            f.write(f'Epoch {epoch}, trainLoss: {avg_train_loss}, evalLoss: {avg_eval_loss},'
                    f'rmse: {rmse}, mae: {mae}, mape: {mape} \n')
            # 在终端输出第多少轮和对应的loss
            print(f'Epoch {epoch}, Train_Loss: {total_loss}')
            if min_mape >mape:
                min_mape = mape
                torch.save(model, args.save_file)  # 保存模型的状态字典到指定文件
        writer.close()


#train()














