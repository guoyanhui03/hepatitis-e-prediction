
from parser_lstm import args
from util import set_seed


from trainLSTM import train       # 单层、双层LSTM的训练函数
#from trainAttLSTM import train  # 单层、双层注意力机制LSTM的训练函数
from trainLSTM_baidu import train # 加百度指数的单层、双层LSTM的训练函数
from trainAttLSTM_baidu import train # 加百度指数的单层、双层注意力机制LSTM的训练函数
# 单因素，双层LSTM参数
args.dataFile = r'./dataset/发病率原始数据.xlsx'   # 修改是发病率还是发病数的数据
args.num_layers =2    # 1表示单层LSTM 2表示双层LSTM
args.input_size =2    # 1表示单因素， 2表示加了百度指数

# 文件夹来控制保存的路径
save_path = r'model/CV_ATT_ST_LSTM_BD_rate/'     # 单因素单层LSTM
#save_path = r'model/CV_ST_LSTM_rate/'     # 单因素堆栈LSTM
for i in range(10):
    set_seed(i)
    save_file = save_path + r'LSTM'+str(i)+r'.pth'
    args.save_file = save_file
    train(args)

