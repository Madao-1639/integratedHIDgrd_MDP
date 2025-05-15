from .trainer import BaseRTFTrainer,BaseTWTrainer,SCTrainer,IntegratedTrainer
from utils.preprocessing import read_preprocess_data
from utils.preprocessing import read_preprocess_data_NoiseAfterScale

def select_trainer(args,**trainer_kwargs):
    if args.model_type == 'Base':
        if args.data_type == 'TW':
            Trainer = BaseTWTrainer
        else:
            Trainer = BaseRTFTrainer
    elif args.model_type == 'SC':
        Trainer = SCTrainer

    if args.k_fold > 0:
        return [Trainer(args,train_data,val_data,**trainer_kwargs) for train_data,val_data in read_preprocess_data(args)]
    elif 0 < args.val_ratio < 1:
        train_data, val_data = read_preprocess_data(args)
        return Trainer(args,train_data,val_data,**trainer_kwargs)
    else:
        data = read_preprocess_data(args)
        return Trainer(args,train_data=data,val_data=None,**trainer_kwargs)



def select_trainer_NoiseAfterScale(args,**trainer_kwargs):
    if args.model_type == 'Base':
        if args.data_type == 'TW':
            Trainer = BaseTWTrainer
        else:
            Trainer = BaseRTFTrainer
    elif args.model_type == 'SC':
        Trainer = SCTrainer

    if args.k_fold > 0:
        return [Trainer(args,train_data,val_data,**trainer_kwargs) for train_data,val_data in read_preprocess_data_NoiseAfterScale(args)]
    elif 0 < args.val_ratio < 1:
        train_data, val_data = read_preprocess_data_NoiseAfterScale(args)
        return Trainer(args,train_data,val_data,**trainer_kwargs)
    else:
        data = read_preprocess_data_NoiseAfterScale(args)
        return Trainer(args,train_data=data,val_data=None,**trainer_kwargs)