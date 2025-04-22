import torch
import torch.nn as nn
from .CustomActvateFunction import select_activate



class BaseTW(nn.Module):
    def __init__(self, args, train_UUTs, mu0=1, sigma0=1, sigma_square=1):
        super().__init__()
        self.lstm = nn.LSTM(
            args.input_size, args.Base_lstm_hidden_size, args.Base_num_lstm_layers,
            dropout=args.Base_lstm_dropout, batch_first=True
            )
        dnn_seq = []
        input_size = args.Base_lstm_hidden_size
        for hidden_size in args.Base_dnn_hidden_sizes:
            dnn_seq.extend([nn.Linear(input_size, hidden_size),nn.ReLU()])
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)
        self.activate = select_activate(args.Base_activate)
        self.cls_thres = args.Base_cls_thres

        # 1ParamBrownian
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(train_UUTs)}
        self.theta_train = nn.Parameter(torch.empty(len(train_UUTs)).normal_(mu0,sigma0))
        self.sigma_square = nn.Parameter(torch.FloatTensor([sigma_square]))

    def forward(self,x,hidden=None):
        '''Input: x (batch_size, window_width, in_features)
        Output:
            hi (batch_size,) -> Health index.
            p (batch_size,) -> Failure probability.'''
        lstm_output, (h,c) = self.lstm(x,hidden)
        hi = self.dnn(h.squeeze(0)).squeeze(-1)
        p = self.activate(hi)
        return hi, p
    
    def predict(self,x,thres=None,**fw_kwargs):
        hi, p = self.forward(x,**fw_kwargs)
        if not thres:
            thres = self.cls_thres
        y_pred = (p>=thres).detach()
        return y_pred
    
    def _UUT2idx(self,UUT):
        if hasattr(UUT,'__getitem__'): # The passed UUT is a sequence
            return [self.train_UUT_dict[_] for _ in UUT]
        else:
            return self.train_UUT_dict[UUT]

    def mfe_loss(self,hi_pre,hi_cur,UUT, reduction = 'none'):
        # Calculate Model Fitting Error(MFE) loss by MLE
        indice = self._UUT2idx(UUT)
        theta = self.theta_train[indice]
        loss = torch.log(self.sigma_square) + torch.square(hi_cur - hi_pre - theta)/self.sigma_square
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

class BaseRTF(BaseTW):
    # def __init__(self, args, train_UUTs, mu0=1, sigma0=1, sigma_square=1):
    #     super().__init__(args, train_UUTs, mu0, sigma0)
    #     self.sigma_square = nn.Parameter(torch.FloatTensor([sigma_square]))

    def forward(self,x,hidden=None):
        lstm_output, (h,c) = self.lstm(x,hidden)
        hi = self.dnn(lstm_output.squeeze(0)).squeeze(-1)
        p = self.activate(hi)
        return hi, p

    def mfe_loss(self,x,UUT, reduction = 'mean'):
        idx = self.train_UUT_dict[UUT]
        theta = self.theta_train[idx]
        n = x.shape[0]
        part1 = torch.log(self.sigma_square)
        part2 = ((x[0] - theta).square() + (x.diff() - theta).square().sum()) / self.sigma_square
        if reduction == 'mean':
            return part1 + part2 / n
        elif reduction == 'sum':
            return n * part1 + part2
        else:
            raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'mean', 'sum'"
        )