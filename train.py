import os

from train import select_trainer
import torch.multiprocessing as mp

from options import prepare_train_args
from utils.utils import set_seed

def main():
    args = prepare_train_args()
    set_seed(args.seed)
    if args.k_fold > 0:
        trainer_seq = select_trainer(args)

        # Multiprocessing CV
        # mp.set_start_method('spawn')
        # pool = mp.Pool(processes = 2)
        for trainer in trainer_seq:
            trainer.train()
        #     pool.apply_async(trainer.train)
        # pool.close()
        # pool.join()
    else:
        trainer = select_trainer(args)
        trainer.train()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set current work directory
    main()