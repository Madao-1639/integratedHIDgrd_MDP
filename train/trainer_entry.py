from .trainer import BaseRTFTrainer,BaseTWTrainer,SCTrainer,IntegratedTrainer
from utils.preprocessing import read_train_data, get_scaler_by_type, LogTransformer, add_noise, gen_cv_data, gen_loo_data

def select_trainer(args,**trainer_kwargs):
    if args.model_type == 'Base':
        if args.data_type == 'TW':
            Trainer = BaseTWTrainer
        else:
            Trainer = BaseRTFTrainer
    elif args.model_type == 'SC':
        Trainer = SCTrainer
    else:
        Trainer = IntegratedTrainer

    # Preprocess data
    data = read_train_data(args.train_fp,args.drop_vars)
    if args.add_noise:
        data = add_noise(data,args.noise_type,args.noise_param)
    if args.k_fold > 0:
        trainer_seq = []
        for i, (train_data, val_data) in enumerate(gen_cv_data(data,args.k_fold)):
            if args.scaler_type:
                scaler = get_scaler_by_type(args.scaler_type)
                train_data.iloc[:,2:] = scaler.fit_transform(train_data.iloc[:,2:])
                val_data.iloc[:,2:] = scaler.transform(val_data.iloc[:,2:])
            if args.log_transform:
                log_transformer = LogTransformer()
                train_data.iloc[:,2:] = log_transformer.fit_transform(train_data.iloc[:,2:])
                val_data.iloc[:,2:] = log_transformer.transform(val_data.iloc[:,2:])
            trainer_seq.append(Trainer(args,train_data,val_data, comment=f'CV{i+1}/{args.k_fold}',**trainer_kwargs))
        return trainer_seq
    elif 0 < args.val_ratio < 1:
        train_data, val_data = gen_loo_data(data,args.val_ratio)
        if args.scaler_type:
            scaler = get_scaler_by_type(args.scaler_type)
            train_data.iloc[:,2:] = scaler.fit_transform(train_data.iloc[:,2:])
            val_data.iloc[:,2:] = scaler.transform(val_data.iloc[:,2:])
        if args.log_transform:
            log_transformer = LogTransformer()
            train_data.iloc[:,2:] = log_transformer.fit_transform(train_data.iloc[:,2:])
            val_data.iloc[:,2:] = log_transformer.transform(val_data.iloc[:,2:])
        return Trainer(args,train_data,val_data,**trainer_kwargs)
    else:
        if args.scaler_type:
            scaler = get_scaler_by_type(args.scaler_type)
            data.iloc[:,2:] = scaler.fit_transform(data.iloc[:,2:])
        if args.log_transform:
            log_transformer = LogTransformer()
            data.iloc[:,2:] = log_transformer.fit_transform(data.iloc[:,2:])
        return Trainer(args,train_data=data,val_data=None,**trainer_kwargs)



def select_trainer_NoiseAfterScale(args,**trainer_kwargs):
    if args.model_type == 'Base':
        if args.data_type == 'TW':
            Trainer = BaseTWTrainer
        else:
            Trainer = BaseRTFTrainer
    elif args.model_type == 'SC':
        Trainer = SCTrainer
    else:
        Trainer = IntegratedTrainer

    # Preprocess data
    data = read_train_data(args.train_fp,args.drop_vars)
    if args.k_fold > 0:
        trainer_seq = []
        for i, (train_data, val_data) in enumerate(gen_cv_data(data,args.k_fold)):
            if args.scaler_type:
                scaler = get_scaler_by_type(args.scaler_type)
                train_data.iloc[:,2:] = scaler.fit_transform(train_data.iloc[:,2:])
                val_data.iloc[:,2:] = scaler.transform(val_data.iloc[:,2:])
            if args.add_noise:
                train_data = add_noise(train_data,args.noise_type,args.noise_param)
                val_data =  add_noise(val_data,args.noise_type,args.noise_param)
            if args.log_transform:
                log_transformer = LogTransformer()
                train_data.iloc[:,2:] = log_transformer.fit_transform(train_data.iloc[:,2:])
                val_data.iloc[:,2:] = log_transformer.transform(val_data.iloc[:,2:])
            trainer_seq.append(Trainer(args,train_data,val_data, comment=f'CV{i+1}/{args.k_fold}',**trainer_kwargs))
        return trainer_seq
    elif 0 < args.val_ratio < 1:
        train_data, val_data = gen_loo_data(data,args.val_ratio)
        if args.scaler_type:
            scaler = get_scaler_by_type(args.scaler_type)
            train_data.iloc[:,2:] = scaler.fit_transform(train_data.iloc[:,2:])
            val_data.iloc[:,2:] = scaler.transform(val_data.iloc[:,2:])
        if args.add_noise:
            train_data = add_noise(train_data,args.noise_type,args.noise_param)
            val_data =  add_noise(val_data,args.noise_type,args.noise_param)
        if args.log_transform:
            log_transformer = LogTransformer()
            train_data.iloc[:,2:] = log_transformer.fit_transform(train_data.iloc[:,2:])
            val_data.iloc[:,2:] = log_transformer.transform(val_data.iloc[:,2:])
        return Trainer(args,train_data,val_data,**trainer_kwargs)
    else:
        if args.scaler_type:
            scaler = get_scaler_by_type(args.scaler_type)
            data.iloc[:,2:] = scaler.fit_transform(data.iloc[:,2:])
        if args.add_noise:
            data = add_noise(data,args.noise_type,args.noise_param)
        if args.log_transform:
            log_transformer = LogTransformer()
            data.iloc[:,2:] = log_transformer.fit_transform(data.iloc[:,2:])
        return Trainer(args,train_data=data,val_data=None,**trainer_kwargs)