import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--dataFile', default='./dataset/发病率原始数据.xlsx')
# 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=200, type=int) # 训练轮数
parser.add_argument('--num_layers', default=2, type=int) # LSTM层数
parser.add_argument('--input_size', default=1, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=50, type=int) # LSTM节点个数
parser.add_argument('--output_size', default=1, type=int) # LSTM输出维度
parser.add_argument('--lr', default=0.001, type=float) #learning rate 学习率
parser.add_argument('--sequence_length', default=4, type=int) # sequence的长度，默认是用前四个月预测下一个月
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--useGPU', default=True, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.15, type=float)
parser.add_argument('--save_file', default='model/LSTM.pth') # 模型保存位置


args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device