'''
tanh(·) is selected as the activation function among the input layer and the hidden layers.
Between the last hidden layer and the output layer, the identity activation function is used.
'''

import torch
import torch.nn as nn
from loss import MVFLoss,MONLoss,CONLoss

class SC_DNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        dnn_seq = []
        input_size = args.input_size
        for hidden_size in args.SC_dnn_hidden_sizes:
            dnn_seq.extend([nn.Linear(input_size, hidden_size),nn.ReLU()])
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)
        # self.dnn = nn.Sequential(
        #     nn.Linear(args.input_size, args.SC_dnn_hidden_size_1),
        #     nn.Tanh(),
        #     nn.Linear(args.SC_dnn_hidden_size_1, args.SC_dnn_hidden_size_2),
        #     nn.Linear(args.SC_dnn_hidden_size_2, 1),
        # )

        # train_UUTs = ls_dict.index
        # self.train_UUT_dict = {UUT:i for i,UUT in enumerate(train_UUTs)}
        # self.MVFLoss_m = args.MVFLoss_m
        # self.MONLoss_c = args.MONLoss_c
        # self.CONLoss_c = args.CONLoss_c
        # self.loss_wa_coef = torch.FloatTensor(1/(ls_dict.values-args.N))

    def forward(self,x):
        return self.dnn(x)

    # def _UUT2idx(self,UUT):
    #     if hasattr(UUT,'__getitem__'): # The passed UUT is a sequence
    #         return [self.train_UUT_dict[_] for _ in UUT]
    #     else:
    #         return self.train_UUT_dict[UUT]

    # def mvf_loss(self,hi_f,UUT, reduction = 'wa'):
    #     if reduction == 'wa':
    #         loss = MVFLoss(hi_f, self.MVFLoss_m, reduction="none")
    #         indice = self._UUT2idx(UUT)
    #         return (self.loss_wa_coef[indice]*loss).sum()
    #     else:
    #         return MVFLoss(hi_f, self.MVFLoss_m, reduction)

    # def mon_loss(self,hi_pre,hi_cur,UUT, reduction = 'wa'):
    #     if reduction == 'wa':
    #         loss = MONLoss(hi_pre,hi_cur, c=self.MONLoss_c, reduction="none")
    #         indice = self._UUT2idx(UUT)
    #         return (self.loss_wa_coef[indice]*loss).sum()
    #     else:
    #         return MONLoss(hi_pre,hi_cur, c=self.MONLoss_c, reduction=reduction)

    # def con_loss(self,hi_ppre,hi_pre,hi_cur,UUT, reduction = 'wa'):
    #     if reduction == 'wa':
    #         loss = CONLoss(hi_ppre,hi_pre,hi_cur, c=self.CONLoss_c, reduction="none")
    #         indice = self._UUT2idx(UUT)
    #         return (self.loss_wa_coef[indice]*loss).sum()
    #     else:
    #         return CONLoss(hi_ppre,hi_pre,hi_cur, c=self.CONLoss_c, reduction=reduction)