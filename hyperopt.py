# import os
import optuna
from train import select_trainer
from options import prepare_train_args
from utils.utils import set_seed


def objective(trial):
    trial.set_user_attr('obj_metric', 'F1')
    # gen_opt_hyperparams
    params = {
        # 'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        'mon_loss_weight': trial.suggest_float('mon_loss_weight', 1, 10),
        'con_loss_weight': trial.suggest_float('con_loss_weight', 1, 10),
    } # Hyperparams to be tuned
    # comment = f'trial{trial.number:04d}' # Add comment in log
    for key, value in params.items():
        setattr(args, key, value) # Update args
        # comment += f'_{key}{value:.2f}'
    trainer = select_trainer(args,\
        opt_trial = trial) #, comment= comment
    return trainer.train()

if __name__ == '__main__':
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = prepare_train_args()
    set_seed(args.seed)
    study = optuna.create_study(
        study_name=args.model_name,
        storage='sqlite:///hyperopt.db',
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
