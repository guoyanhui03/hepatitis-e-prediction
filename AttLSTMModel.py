import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AttLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = nn.Linear(4, 4)  # 与时间步的值相同

    def forward(self, x):
        x = x.permute(0,2,1)
        att = self.attention(x)
        att = F.softmax(att, dim=-1)
        x = torch.mul(x, att)
        x = x.permute(0,2,1)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output