'''
tanh(·) is selected as the activation function among the input layer and the hidden layers.
Between the last hidden layer and the output layer, the identity activation function is used.
'''

# import torch
import torch.nn as nn
from loss import MVFLoss,MONLoss,CONLoss

class SC_DNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        dnn_seq = []
        input_size = args.input_size
        for hidden_size in args.SC_dnn_hidden_sizes:
            dnn_seq.append(nn.Linear(input_size, hidden_size))
            dnn_seq.append(nn.ReLU())
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)

    def forward(self,x):
        return self.dnn(x)