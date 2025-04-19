from torch.utils.data import Dataset,IterableDataset
import torch

class BaseDataset(Dataset):
    def __init__(self, data, train, args):
        ''' Base `Dataset` class for CMAPSS FD001 Dataset.
        Input a Dataframe with columns ('UUT','time',...variables).'''
        super().__init__()
        # self.data = data.copy()
        self.data = data
        self.train = train
        self.var_columns = self.data.columns[2:]

        self.task = args.task
        self.get_label()

        grouped = self.data.groupby('UUT')
        self.grouped = grouped
        self.ls_dict = grouped['time'].max()

        self.drop_failure = args.drop_failure
        self.get_sample_indice()

    def get_label(self):
        failure_time = self.data.groupby('UUT')['time'].transform('max')
        if self.task == 'cls':
            def _assign_label(x):
                if x == 0:
                    return -1
                elif x == 1:
                    return 1
                elif x > 0:
                    return 0
            self.data = self.data.assign(label=(failure_time-self.data['time']).map(_assign_label))
        else:
            self.data = self.data.assign(label = failure_time-self.data['time'])

    def get_sample_indice(self):
        if self.drop_failure:
            if self.task == 'cls':
                self.sample_indice = self.data[self.data['label']>=0].index
            else:
                self.sample_indice = self.data[self.data['label']>0].index
        else:
            self.sample_indice = self.data.index

    def __getitem__(self, index):
        idx = self.sample_indice(index)
        UUT, t, *data, y = self.data.loc[[idx]]
        return UUT, t, data, y

    def __len__(self):
        return len(self.sample_indice)



class BaseDataset_ND(BaseDataset):
    def __init__(self, data, train, args):
        self.N = args.N
        super().__init__(data,train,args)

    def get_sample_indice(self):
        sample_indice = []
        diff = int(self.drop_failure)
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT] - diff
            sample_indice.extend(
                (UUT,t,
                *(group_indice[t-i] for i in range(self.N+1,0,-1)),
                self.data.loc[group_indice[t-1],'label']    # Label for current time.
                ) for t in range(self.N+1,end_time+1)
            )
        self.sample_indice = sample_indice

    def __getitem__(self, index):        
        UUT, t, *indice, y = self.sample_indice[index]
        return UUT, t, *(self.data.loc[idx,self.var_columns].astype('float32') for idx in indice), y



class BaseDataset_ND_F(BaseDataset_ND):
    def get_sample_indice(self):
        sample_indice = []
        diff = int(self.drop_failure)
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT] - diff
            sample_indice.extend(
                (UUT,t,
                *(group_indice[t-i] for i in range(self.N+1,0,-1)),
                group_indice[-1],
                self.data.loc[group_indice[t-1],'label']
                ) for t in range(self.N+1,end_time+1)
            )
        self.sample_indice = sample_indice



class TWDataset(BaseDataset):
    def __init__(self, data, train, args):
        ''' Input: data (Dataframe)
        Create Time Windows(TW) of data.
        A window, which is an element of one minibatch, has 3 elements:
            UUT, t (time series the window covers) and x (features),
            or 5 elements for train data, with y (labels) and start_sign (A bool to indicate the first window) added.
        There is no overlap if 'sliding_offset' >= 'window_width'.'''
        self.window_width = args.window_width
        super().__init__(data, train,args)
    
    def get_sample_indice(self):
        '''Create Time Windows(TW) of data.
        Return a list of (UUT,time,start_idx,end_idx) indicating a window.'''
        sample_indice = []
        diff = int(self.drop_failure)
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT] - diff
            sample_indice.extend(
                (UUT,t,
                    group_indice[t-self.window_width:t], # Window indice.
                    self.data.loc[group_indice[t-1],'label'] # Label for current window.
                ) for t in range(self.window_width,end_time+1)
            )
        self.sample_indice = sample_indice
    
    def __getitem__(self, index):
        UUT, t, indice, y =  self.sample_indice[index]
        data = self.data.loc[indice,self.var_columns]  # Current window
        return UUT, t, data, y



class TWDataset_ND(TWDataset):
    def __init__(self, data, train, args):
        self.N = args.N
        super().__init__(data,train,args)

    def get_sample_indice(self):
        sample_indice = []
        diff = int(self.drop_failure)
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT] - diff
            sample_indice.extend(
                (UUT,t,
                    *(group_indice[t-i-self.window_width:t-i]
                    for i in range(self.N,-1,-1)),
                    self.data.loc[group_indice[t-1],'label']
                ) for t in range(self.window_width+self.N,end_time+1)
            )
        self.sample_indice = sample_indice

    def __getitem__(self, index):        
        UUT, t, *indice_tuple, y = self.sample_indice[index]
        return UUT, t, *(self.data.loc[indice,self.var_columns] for indice in indice_tuple), y



class TWDataset_ND_F(TWDataset_ND):
    def get_sample_indice(self):
        sample_indice = []
        diff = int(self.drop_failure)
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT] - diff
            sample_indice.extend(
                (UUT,t,
                    *(group_indice[t-i-self.window_width:t-i]
                    for i in range(self.N,-1,-1)),
                    group_indice[-self.window_width:], # Failure window indice.
                    self.data.loc[group_indice[t-1],'label']
                ) for t in range(self.window_width+self.N,end_time+1)
            )
        self.sample_indice = sample_indice



class RTFDataset(IterableDataset):
    def __init__(self, data, train, args):
        super().__init__()
        self.data = data
        self.train = train
        self.var_columns = self.data.columns[2:]

        self.task = args.task
        self.get_label()

        grouped = self.data.groupby('UUT')
        self.grouped = grouped
        self.ls_dict = grouped['time'].max()

        self.drop_failure = args.drop_failure

    def get_label(self):
        failure_time = self.data.groupby('UUT')['time'].transform('max')
        if self.task == 'cls':
            def _assign_label(x):
                if x == 0:
                    return -1
                elif x == 1:
                    return 1
                elif x > 0:
                    return 0
            self.data = self.data.assign(label=(failure_time-self.data['time']).map(_assign_label))
        else:
            self.data = self.data.assign(label = failure_time-self.data['time'])

    def __iter__(self):
        diff = int(self.drop_failure)
        for UUT, grouped_data in self.grouped:
            end_time = self.ls_dict[UUT] - diff
            grouped_data = grouped_data.iloc[:end_time]
            t = grouped_data['time']
            y = grouped_data['label']
            data = grouped_data[self.var_columns]
            yield UUT, t, data, y



class RTFTWDataset(RTFDataset):
    def __init__(self, data, train, args):
        self.window_width = args.window_width 
        super().__init__(data, train, args)

    def __iter__(self):
        diff = int(self.drop_failure)
        for UUT, grouped_data in self.grouped:
            end_time = self.ls_dict[UUT] - diff
            grouped_aux = grouped_data.iloc[self.window_width-1:end_time]
            t = grouped_aux['time']
            y = grouped_aux['label']
            data = []
            for window_time in range(self.window_width,end_time+1):
                window_data = grouped_data.iloc[window_time-self.window_width:window_time,2:-1]
                data.append(window_data)
            yield UUT, t, data, y