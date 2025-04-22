import torch
from torch import nn
from torch.optim import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest,shapiro,normaltest,anderson
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

from utils.data import select_loader
from model import BaseTW, BaseRTF, SC_DNN
from loss import FocalLoss,MVFLoss,MONLoss,CONLoss
from utils.logger import Logger

from sklearn.linear_model import LogisticRegression

class BaseTWTrainer:
    '''Train BaseTW Model'''
    def __init__(self,args,train_data,val_data=None,**logger_kwargs):
        self.args = args
        self.logger = Logger(args,**logger_kwargs)
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f'Training on {self.device}')

        self.get_loader(train_data,val_data)
        self.get_model()
        if args.loss_wa:
            self.get_loss_wa_coef()
        self.get_optimizer()

    def get_loader(self,train_data,val_data):
        self.train_loader = select_loader(train_data,True,self.args)
        self.ls_dict = self.train_loader.dataset.ls_dict

        if val_data is not None:
            self.val_loader = select_loader(val_data,False,self.args)
        else:
            self.val_loader = None

        if self.args.record_HI:
            if self.args.record_HI == 'train':
                record_HI_data = train_data
            elif self.args.record_HI == 'val':
                record_HI_data = val_data
            else:
                record_HI_data = pd.concat([train_data,val_data])
            if self.args.data_type == 'TW':
                record_HI_data_type = 'RTFTW'
            else:
                record_HI_data_type = 'RTF'
            self.record_HI_loader = select_loader(record_HI_data,False,self.args,record_HI_data_type)
            data_record_UUTs = record_HI_data['UUT'].unique()
            args_record_UUTs = self.args.record_UUTs
            if args_record_UUTs and all((data_record_UUTs==UUT).any() for UUT in args_record_UUTs):
                self.record_UUTs = args_record_UUTs
            else:
                self.record_UUTs = np.random.choice(data_record_UUTs,size=5,replace=False)
        else:
            self.record_HI_loader = None

    def get_model(self):
        self.model = BaseTW(self.args, self.train_loader.dataset.ls_dict.index)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_path).state_dict())
        self.model.to(self.device)
        # example_input = torch.randn((self.args.window_width,self.args.input_size,))
        # self.logger.writer.add_graph(self.model,example_input)

    def get_loss_wa_coef(self):
        # Coefficients of Weighted-Average(WA) of loss for each UUT.
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(self.ls_dict.index)}
        self.loss_wa_coef = torch.FloatTensor(1/(self.ls_dict.values-self.args.window_width-self.args.N))

    def _UUT2idx(self,UUT):
        if hasattr(UUT,'__getitem__'): # The passed UUT is a sequence
            return [self.train_UUT_dict[_] for _ in UUT]
        else:
            return self.train_UUT_dict[UUT]

    def get_optimizer(self):
        no_decay_pg, decay_pg = [], []
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm1d):
                no_decay_pg.extend(layer.parameters())
            else:
                for name, param in layer.named_parameters(recurse=False):
                    if 'weight' in name:
                        decay_pg.append(param)
                    else:
                        no_decay_pg.append(param)
        optimizer = Adam([{'params':no_decay_pg}], lr=self.args.lr)
        optimizer.add_param_group({'params': decay_pg, 'weight_decay': self.args.weight_decay})
        self.optimizer = optimizer

    def train(self):
        for epoch in range(1,self.args.num_epoch+1):
            self.train_per_epoch(epoch)
            if self.val_loader is not None:
                self.val_per_epoch(epoch)
            if self.record_HI_loader is not None:
                self.record_per_epoch(epoch)
            self.logger.save_metrics(epoch)
            self.logger.save_checkpoint(self.model, epoch)

    def train_per_epoch(self, epoch):
        # Switch to train mode
        self.model.train()

        for i, (UUT, t, X_pre, X_cur, y_true) in enumerate(self.train_loader):
            X_pre = X_pre.to(self.device)
            X_cur = X_cur.to(self.device)
            y_true = y_true.to(self.device)

            hi_pre, _ = self.model(X_pre)
            hi_cur, p = self.model(X_cur)

            # Compute loss
            loss = self.compute_loss(hi_pre, hi_cur, p, y_true, UUT)

            # Get the item for backward
            total_loss = loss['total_loss']

            # Compute gradient and do Adam step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logger record loss
            for k,v in loss.items():
                self.logger.record_scalars('Loss/train', k, v)

            # Monitor training progress
            if (i+1) % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i+1} Loss {total_loss.item():.6f}')

    def val_per_epoch(self, epoch):
        self.model.eval()

        # Classification metrics
        metric_dict = {
            'Recall': recall_score,
            'Precision': precision_score,
            'F1': f1_score,
        }
        all_y_true = []
        all_y_pred = []
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            y_pred = self.model.predict(X)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
        all_y_true = torch.concat(all_y_true)
        all_y_pred = torch.concat(all_y_pred)
        for metric_name,metric_fun in metric_dict.items():
            metric = metric_fun(all_y_true,all_y_pred)
            self.logger.writer.add_scalar(f'{metric_name}/val',metric,epoch)

        self.logger.record_histogram('theta/train',self.model.theta_train)

    def record_per_epoch(self,epoch):
        self.model.eval()

        hi_dict = {}
        # Test for normality
        sig_list = sorted((0.01,0.05,0.10))
        nt_summary = {} # 'epoch':epoch
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi,p = self.model(X)
            hi = hi.detach().numpy()
            hi_dict[UUT] = hi

            resInc = np.diff(hi)
            nt_result = {
                'KS': kstest(resInc,cdf='norm'),
                'SW': shapiro(resInc),
                'DP': normaltest(resInc),
                'AD': anderson(resInc,dist='norm')
            }
            for test_name, test_result in nt_result.items():
                if test_name == 'AD':
                    for sig in sig_list:
                        ad_idx = np.where(test_result.significance_level==sig*100)[0][0]
                        critical_value = test_result.critical_values[ad_idx]
                        statistic = test_result.statistic
                        if statistic < critical_value:
                            nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})',0) + 1
                        else:
                            break
                else:
                    statistic, p_value = test_result
                    for sig in sig_list:
                        if p_value > sig:
                            nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})',0) + 1
                        else:
                            break
        for k,v in nt_summary.items():
            self.logger.record_scalars(f'NomalTest/{self.args.record_HI}',k,v)

        # Plot selected HI
        if epoch % self.args.record_freq == 0:
            fig = plt.figure(figsize=(10,5))
            for UUT in self.record_UUTs:
                hi = hi_dict[UUT]
                plt.plot(hi,'-',lw=0.5,alpha=0.75,label=UUT)
            plt.legend()
            # plt.title(f'epoch:{epoch}')
            plt.tight_layout()
            self.logger.writer.add_figure(f'HI/{self.args.record_HI}',fig,epoch)

    def compute_loss(self, hi_pre, hi_cur, p, y_true, UUT):
        if self.args.loss_wa:
            indice = self._UUT2idx(UUT)
            loss_wa_coef = self.loss_wa_coef[indice]
        
            cls_loss = self.args.cls_loss_weight * (loss_wa_coef@FocalLoss(p, y_true, alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction='none'))
            mfe_loss = self.args.mfe_loss_weight * (loss_wa_coef@self.model.mfe_loss(hi_pre, hi_cur, UUT, reduction = 'none'))
        else:
            cls_loss = self.args.cls_loss_weight * FocalLoss(p, y_true, alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction='mean')
            mfe_loss = self.args.mfe_loss_weight * self.model.mfe_loss(hi_pre, hi_cur, UUT, reduction = 'mean')

        total_loss = cls_loss + mfe_loss
        loss = {
            'cls_loss': cls_loss,
            'mfe_loss': mfe_loss,
            'total_loss': total_loss
        }
        return loss



class BaseRTFTrainer(BaseTWTrainer):
    def get_model(self):
        self.model = BaseRTF(self.args, self.train_loader.dataset.ls_dict.index)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_path).state_dict())
        self.model.to(self.device)
        # example_input = torch.randn((self.args.input_size,20))
        # self.logger.writer.add_graph(self.model,example_input)

    def train_per_epoch(self, epoch):
        # Switch to train mode
        self.model.train()

        for i, (UUT, t, X, y_true) in enumerate(self.train_loader):
            X = X.to(self.device)
            y_true = y_true.to(self.device)

            hi, p = self.model(X)

            # Compute loss
            loss = self.compute_loss(hi, p, y_true, UUT)

            # Get the item for backward
            total_loss = loss['total_loss']

            # Compute gradient and do Adam step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logger record loss
            for k,v in loss.items():
                self.logger.record_scalars('Loss/train', k, v)

            # Monitor training progress
            if (i+1) % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i+1} Loss {total_loss.item():.6f}')

    def record_per_epoch(self,epoch):
        self.model.eval()

        hi_dict = {}
        # Test for normality
        sig_list = sorted((0.01,0.05,0.10))
        nt_summary = {} # 'epoch':epoch
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi,p = self.model(X)
            hi = hi.detach().numpy()
            hi_dict[UUT] = hi

            resInc = np.diff(hi,prepend=0)[10:]
            nt_result = {
                'KS': kstest(resInc,cdf='norm'),
                'SW': shapiro(resInc),
                'DP': normaltest(resInc),
                'AD': anderson(resInc,dist='norm')
            }
            for test_name, test_result in nt_result.items():
                if test_name == 'AD':
                    for sig in sig_list:
                        ad_idx = np.where(test_result.significance_level==sig*100)[0][0]
                        critical_value = test_result.critical_values[ad_idx]
                        statistic = test_result.statistic
                        if statistic < critical_value:
                            nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})',0) + 1
                        else:
                            break
                else:
                    statistic, p_value = test_result
                    for sig in sig_list:
                        if p_value > sig:
                            nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})',0) + 1
                        else:
                            break
        for k,v in nt_summary.items():
            self.logger.record_scalars(f'NomalTest/{self.args.record_HI}',k,v)

        # Plot selected HI
        if epoch % self.args.record_freq == 0:
            fig = plt.figure(figsize=(10,5))
            for UUT in self.record_UUTs:
                hi = hi_dict[UUT]
                plt.plot(hi,'-',lw=0.5,alpha=0.75,label=UUT)
            plt.legend()
            # plt.title(f'epoch:{epoch}')
            plt.tight_layout()
            self.logger.writer.add_figure(f'HI/{self.args.record_HI}',fig,epoch)

    def compute_loss(self, hi, p, y_true, UUT):
        cls_loss = self.args.cls_loss_weight * FocalLoss(p, y_true, alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction='mean')
        mfe_loss = self.args.mfe_loss_weight * self.model.mfe_loss(hi, UUT, reduction = 'mean')

        total_loss = cls_loss + mfe_loss
        loss = {
            'cls_loss': cls_loss,
            'mfe_loss': mfe_loss,
            'total_loss': total_loss
        }
        return loss

class SCTrainer(BaseTWTrainer):
    def get_model(self):
        self.model = SC_DNN(self.args)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_path).state_dict())
        self.model.to(self.device)
        example_input = torch.randn((self.args.input_size,))
        self.logger.writer.add_graph(self.model,example_input)

    def get_loss_wa_coef(self):
        # Coefficients of Weighted-Average(WA) of loss for each UUT.
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(self.ls_dict.index)}
        self.loss_wa_coef = torch.FloatTensor(1/(self.ls_dict.values-self.args.N))

    def train_per_epoch(self, epoch):
        # Switch to train mode
        self.model.train()

        all_y_true = []
        all_hi = []
        for i, (UUT, t, X_ppre, X_pre, X_cur, X_f, y_true) in enumerate(self.train_loader):
            X_ppre = X_ppre.to(self.device)
            X_pre = X_pre.to(self.device)
            X_cur = X_cur.to(self.device)
            X_f = X_f.to(self.device)
            y_true = y_true.to(self.device)

            hi_ppre = self.model(X_ppre)
            hi_pre = self.model(X_pre)
            hi_cur = self.model(X_cur)
            hi_f = self.model(X_f)

            # Compute loss
            loss = self.compute_loss(hi_ppre, hi_pre, hi_cur, hi_f, UUT)

            # Get the item for backward
            total_loss = loss['total_loss']

            # Compute gradient and do Adam step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logger record
            for k,v in loss.items():
                self.logger.record_scalars('Loss/train', k, v)

            # Monitor training progress
            if (i+1) % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i+1} Loss {total_loss.item():.6f}')
            
            all_y_true.append(y_true)
            all_hi.append(hi_cur.detach())
        all_y_true = torch.concat(all_y_true)
        all_hi = torch.concat(all_hi)
        self.cls_model = LogisticRegression(class_weight='balanced',)
        self.cls_model.fit(all_hi,all_y_true)

    def val_per_epoch(self, epoch):
        self.model.eval()

        # Classification metrics
        metric_dict = {
            'Recall': recall_score,
            'Precision': precision_score,
            'F1': f1_score,
        }
        all_y_true = []
        all_hi = []
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi = self.model(X).detach()
            all_y_true.append(y_true)
            all_hi.append(hi)
        all_y_true = torch.concat(all_y_true)
        all_hi = torch.concat(all_hi)
        all_y_pred = self.cls_model.predict(all_hi)
        for metric_name,metric_fun in metric_dict.items():
            metric = metric_fun(all_y_true,all_y_pred)
            self.logger.writer.add_scalar(f'{metric_name}/val',metric,epoch)

    def record_per_epoch(self,epoch):
        self.model.eval()

        # Plot selected HI
        if epoch % self.args.record_freq == 0:
            fig = plt.figure(figsize=(10,5))
            for UUT,t,X,y_true in self.record_HI_loader:
                if UUT in self.record_UUTs:
                    X = X.to(self.device)
                    hi = self.model(X).detach().numpy()
                    plt.plot(hi,'-',lw=0.5,alpha=0.75,label=UUT)
            plt.legend()
            # plt.title(f'epoch:{epoch}')
            plt.tight_layout()
            self.logger.writer.add_figure(f'HI/{self.args.record_HI}',fig,epoch)

    def compute_loss(self, hi_ppre, hi_pre, hi_cur, hi_f, UUT):
        indice = self._UUT2idx(UUT)
        loss_wa_coef = self.loss_wa_coef[indice]

        mvf_loss = self.args.mvf_loss_weight * (loss_wa_coef@MVFLoss(hi_f, self.args.MVFLoss_m, reduction="none"))
        mon_loss = self.args.mon_loss_weight * (loss_wa_coef@MONLoss(hi_pre,hi_cur, c=self.args.MONLoss_c, reduction="none"))
        con_loss = self.args.con_loss_weight * (loss_wa_coef@CONLoss(hi_ppre,hi_pre,hi_cur, c=self.args.CONLoss_c, reduction="none"))
        total_loss = mvf_loss + mon_loss + con_loss

        loss = {
            'mvf_loss': mvf_loss,
            'con_loss': con_loss,
            'mon_loss': mon_loss,
            'total_loss': total_loss
        }
        return loss