import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

class Integrated_DNN_LSTM(nn.Module):
    def __init__(self,args, train_UUTs, mu0=1, sigma0=1):
        '''Run-to-failure'''
        super().__init__()
        # HI construction
        dnn_seq = []
        input_size = args.input_size
        for hidden_size in args.Integrated_dnn_hidden_sizes:
            dnn_seq.append(nn.Linear(input_size, hidden_size))
            dnn_seq.append(nn.Softplus())
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)

        # Degradation model
        self.tmax = 500
        self.lstm = nn.LSTM(
            1, args.Integrated_lstm_hidden_size, args.Integrated_num_lstm_layers,
            dropout=args.Integrated_lstm_dropout, batch_first=True
        )
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(train_UUTs)}
        self.Gamma_train = nn.Parameter(torch.empty(len(train_UUTs),args.Integrated_lstm_hidden_size).normal_(mu0,sigma0))

        self.cls_model = LogisticRegression(class_weight='balanced',)

    def forward(self,x):
        return self.dnn(x).squeeze(-1)  # Degradation signals / HI

    def mfe_loss(self,dgrd_signal,t,UUT, reduction = 'mean'):
        idx = self.train_UUT_dict[UUT]
        Gamma = self.Gamma_train[idx]
        psi,(_,_) = self.lstm(t.unsqueeze(1)/self.tmax)
        dgrd_status = psi@Gamma
        loss = (dgrd_signal - dgrd_status).square()
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'mean', 'sum'"
        )

    def fit(self,hi,y):
        self.cls_model.fit(hi.reshape(-1, 1),y)

    def predict(self,X):
        hi = self.forward(X).detach()
        return self.cls_model.predict(hi.reshape(-1, 1))