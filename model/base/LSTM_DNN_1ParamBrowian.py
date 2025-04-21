import torch
import torch.nn as nn
import torch.nn.functional as F
# from loss import FocalLoss

class SigmoidExpBias(nn.Module):
    def __init__(self,bias=0):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([bias]))
    
    def forward(self,x):
        return F.sigmoid(torch.exp(x)+self.bias)



class SigmoidLinear(nn.Module):
    def __init__(self,weight=None,bias=0):
        super().__init__()
        if weight is None:
            self.weight = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.uniform_(self.weight)
        else:
            self.weight = nn.Parameter(torch.FloatTensor([weight]))
        self.bias = nn.Parameter(torch.FloatTensor([bias]))
    
    def forward(self,x):
        return F.sigmoid(self.weight*x+self.bias)



def SigmoidLeakyReLU(x,alpha=0.01):
    return F.sigmoid(F.leaky_relu(x,alpha))



class SigmoidLinearReLU(nn.Module):
    def __init__(self,weight=None,bias=0):
        super().__init__()
        if weight is None:
            self.weight = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.uniform_(self.weight)
        else:
            self.weight = nn.Parameter(torch.FloatTensor([weight]))
        self.bias = nn.Parameter(torch.FloatTensor([bias]))
    
    def forward(self,x):
        return F.sigmoid(F.relu(self.weight*x+self.bias))



def SigmoidELU(x,alpha=0.01):
    return F.sigmoid(F.elu(x,alpha))



class LSTM_DNN_1ParamBrowian(nn.Module):
    def __init__(self, args, train_UUTs, mu0=0.1, sigma0=0.1, sigma_square = 1):
        super().__init__()
        self.data_type = args.data_type
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
        
        if args.Base_activate == 'SigmoidExpBias':
            self.activate = SigmoidExpBias()
        elif args.Base_activate == 'SigmoidLinear':
            self.activate = SigmoidLinear()
        elif args.Base_activate == 'SigmoidLinearReLU':
            self.activate = SigmoidLinearReLU()
        elif args.Base_activate == 'SigmoidLeakyReLU':
            self.activate = SigmoidLeakyReLU
        elif args.Base_activate == 'SigmoidELU':
            self.activate = SigmoidELU
        else:
            self.activate = F.sigmoid
        self.cls_thres = args.Base_cls_thres

        # 1ParamBrownian
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(train_UUTs)}
        self.theta_train = nn.Parameter(torch.empty(len(train_UUTs)).normal_(mu0,sigma0))
        self.sigma_square = nn.Parameter(torch.FloatTensor([sigma_square]))

        # Loss
        # self.FocalLoss_alpha = args.FocalLoss_alpha
        # self.FocalLoss_gamma = args.FocalLoss_gamma
        # self.loss_wa_coef = torch.FloatTensor(1/(ls_dict.values-args.window_width))

    def forward(self,x,hidden=None):
        '''Input: x (batch_size, window_width, in_features)
        Output:
            hi (batch_size,) -> Health index.
            p (batch_size,) -> Failure probability.'''
        lstm_output, (h,c) = self.lstm(x,hidden)
        if self.data_type == 'TW':
            hi = self.dnn(h.squeeze(0)).squeeze(-1)
        elif self.data_type == 'Base':
            hi = self.dnn(lstm_output.squeeze(0)).squeeze(-1)
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
        # loss = torch.log(self.sigma_square + 1e-7) + torch.square(hi_cur - hi_pre - theta)/self.sigma_square
        loss = torch.square(hi_cur - hi_pre - theta)
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
    
    # def cls_loss(self, p, y_true, UUT=None, reduction='wa'):
    #     if reduction == 'wa':
    #         indice = self._UUT2idx(UUT)
    #         loss = FocalLoss(p, y_true, self.FocalLoss_alpha, self.FocalLoss_gamma, reduction='none')
    #         return (self.loss_wa_coef[indice]*loss).sum()
    #     else:
    #         return FocalLoss(p, y_true, self.FocalLoss_alpha, self.FocalLoss_gamma, reduction)