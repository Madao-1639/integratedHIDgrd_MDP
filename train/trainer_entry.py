from .trainer import BaseRTFTrainer,BaseTWTrainer,SCTrainer,IntegratedTrainer,MSRTFTrainer#,MSTWTrainer
from utils.preprocessing import read_preprocess_data
from utils.preprocessing import read_preprocess_data_NoiseAfterScale

def get_trainer_by_type(model_type, data_type=None):
    if model_type == 'Base':
        if data_type == 'RTF':
            return BaseRTFTrainer
        elif data_type == 'TW':
            return BaseTWTrainer
    elif model_type == 'SC':
        return SCTrainer
    elif model_type == 'Integrated':
        return IntegratedTrainer
    elif model_type == 'MS':
        if data_type == 'RTF':
            return MSRTFTrainer
        # elif data_type == 'TW':
        #     return MSTWTrainer

def select_trainer(args,**trainer_kwargs):
    Trainer = get_trainer_by_type(args.model_type,args.data_type)
    if args.k_fold > 0:
        return [Trainer(args,train_data,val_data,**trainer_kwargs) for train_data,val_data in read_preprocess_data(args)]
    elif 0 < args.val_ratio < 1:
        train_data, val_data = read_preprocess_data(args)
        return Trainer(args,train_data,val_data,**trainer_kwargs)
    else:
        data = read_preprocess_data(args)
        return Trainer(args,train_data=data,val_data=None,**trainer_kwargs)



def select_trainer_NoiseAfterScale(args,**trainer_kwargs):
    Trainer = get_trainer_by_type(args.model_type,args.data_type)
    if args.k_fold > 0:
        return [Trainer(args,train_data,val_data,**trainer_kwargs) for train_data,val_data in read_preprocess_data_NoiseAfterScale(args)]
    elif 0 < args.val_ratio < 1:
        train_data, val_data = read_preprocess_data_NoiseAfterScale(args)
        return Trainer(args,train_data,val_data,**trainer_kwargs)
    else:
        data = read_preprocess_data_NoiseAfterScale(args)
        return Trainer(args,train_data=data,val_data=None,**trainer_kwargs)