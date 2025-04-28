from train import select_trainer
from options import prepare_train_args
from utils.utils import set_seed

def main():
    args = prepare_train_args()
    set_seed(args.seed)
    if args.k_fold > 0:
        trainer_seq = select_trainer(args)
        for trainer in trainer_seq:
            trainer.train()
    else:
        trainer = select_trainer(args)
        trainer.train()

if __name__ == '__main__':
    main()