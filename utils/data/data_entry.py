from .dataset import (
    BaseDataset,
    BaseDataset_ND,
    TWDataset,
    TWDataset_ND,
    RTFDataset,
    RTFTWDataset,
)
import torch
from torch.utils.data import DataLoader

def _pack_data(batch_data):
    return torch.stack([torch.FloatTensor(data) for data in batch_data])

def custom_collate_fn(batch):
    batch_UUT,batch_t,batch_X, y = zip(*batch)
    batch_t = torch.FloatTensor(batch_t)
    y = torch.FloatTensor(y)
    return batch_UUT,batch_t, _pack_data(batch_X), y

def custom_collate_fn_ND(batch):
    batch_start, batch_end, batch_UUT, batch_t, batch_multi_X, batch_multi_y = zip(*batch)
    batch_start = torch.BoolTensor(batch_start)
    batch_end = torch.BoolTensor(batch_end)
    batch_t = torch.FloatTensor(batch_t)
    return batch_start, batch_end, batch_UUT,batch_t, \
        list(_pack_data(batch_X) for batch_X in zip(*batch_multi_X)), \
        list(torch.FloatTensor(batch_y) for batch_y in zip(*batch_multi_y))

def custom_RTF_collate_fn(batch):
    UUT,t,data, y = batch[0]
    t = torch.FloatTensor(t)
    y = torch.FloatTensor(y)
    if isinstance(data,list):
        data = _pack_data(data)
    else:
        data = torch.FloatTensor(data)
    return UUT,t, data, y

def select_loader(data,train,args,data_type=None):
    if data_type == None:
        data_type = args.data_type
    if data_type == 'Base':
        if train and args.N > 0:
            dataset = BaseDataset_ND(data,train,args)
            dataloader = DataLoader(dataset, args.batch_size,
            shuffle=train, num_workers=args.num_workers, collate_fn=custom_collate_fn_ND, pin_memory=True, drop_last=False)
        else:
            dataset = BaseDataset(data,train,args)
            dataloader = DataLoader(dataset, args.batch_size,
            shuffle=train, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True, drop_last=False)
    elif data_type == 'TW':
        if train and args.N > 0:
            dataset = TWDataset_ND(data,train,args)
            dataloader = DataLoader(dataset, args.batch_size,
            shuffle=train, num_workers=args.num_workers, collate_fn=custom_collate_fn_ND, pin_memory=True, drop_last=False)
        else:
            dataset = TWDataset(data,train,args)
            dataloader = DataLoader(dataset, args.batch_size,
            shuffle=train, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True, drop_last=False)
    elif data_type == 'RTF':
        dataset = RTFDataset(data,train,args)
        dataloader = DataLoader(dataset,collate_fn=custom_RTF_collate_fn,pin_memory=True,)
    elif data_type == 'RTFTW':
        dataset = RTFTWDataset(data,train,args)
        dataloader = DataLoader(dataset,collate_fn=custom_RTF_collate_fn,pin_memory=True,)
    else:
        raise ValueError(
            f"Invalid Value for arg 'data_type': '{data_type} \n Supported data_type: 'Base', 'TW', 'RTF', 'RTFTW'"
        )
    return dataloader