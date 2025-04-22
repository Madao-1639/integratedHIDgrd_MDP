from .dataset import (
    BaseDataset,
    BaseDataset_ND,
    BaseDataset_ND_F,
    TWDataset,
    TWDataset_ND,
    TWDataset_ND_F,
    RTFDataset,
    RTFTWDataset,
)
import torch

from torch.utils.data import DataLoader

def get_dataset(data_type,data,train,args,):
    if data_type == 'Base':
        if train:
            if args.F:
                return BaseDataset_ND_F(data,train,args)
            elif args.N > 0:
                return BaseDataset_ND(data,train,args)
            else:
                return BaseDataset(data,train,args)
        else:
            return BaseDataset(data,train,args)
    elif data_type == 'TW':
        if train:
            if args.F:
                return TWDataset_ND_F(data,train,args)
            elif args.N > 0:
                return TWDataset_ND(data,train,args)
            else:
                return TWDataset(data,train,args)
        else:
            return TWDataset(data,train,args)
    elif data_type == 'RTF':
        return RTFDataset(data,train,args)
    elif data_type == 'RTFTW':
        return RTFTWDataset(data,train,args)
    else:
        raise ValueError(
            f"Invalid Value for arg 'data_type': '{data_type} \n Supported data_type: 'Base', 'TW', 'RTF', 'RTFTW'"
        )

def _pack_data(batch_data):
    return torch.stack([torch.from_numpy(data.values) for data in batch_data])

def custom_collate_fn(batch):
    batch_UUT,batch_t,*batch_multidata, y = zip(*batch)
    batch_t = torch.FloatTensor(batch_t)
    y = torch.FloatTensor(y)
    return batch_UUT,batch_t,*(_pack_data(batch_data) for batch_data in batch_multidata), y

def custom_RTF_collate_fn(batch):
    UUT,t,data, y = batch[0]
    t = torch.FloatTensor(t.values)
    y = torch.FloatTensor(y.values)
    if isinstance(data,list):
        data = _pack_data(data)
    else:
        data = torch.FloatTensor(data.values)
    return UUT,t, data, y

def select_loader(data,train,args,data_type=None):
    if not data_type:
        data_type = args.data_type
    dataset = get_dataset(data_type, data, train, args)
    if data_type in ('RTF','RTFTW'):
        loader = DataLoader(dataset,collate_fn=custom_RTF_collate_fn,pin_memory=True,)
        
    else:
        loader = DataLoader(dataset, args.batch_size,
            shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True, drop_last=False)
    return loader