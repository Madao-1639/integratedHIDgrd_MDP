import os
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

# https://pytorch.org/docs/stable/tensorboard.html
# https://m.w3cschool.cn/article/27419536.html


# class Recorder:
#     def __init__(self):
#         self.scalar_dict = defaultdict(list)
#         self.scalars_dict = defaultdict(lambda: defaultdict(list))
#         self.histogram_dict = defaultdict(list)
#         self.hi_dict = {}

#     def record_scalar(self, tag, value):
#         self.scalar_dict[tag].append(value)
    
#     def record_scalars(self, main_tag, tag, value):
#         self.scalars_dict[main_tag][tag].append(value)
    
#     def record_histogram(self, tag, values):
#         self.histogram_dict[tag].append(values)

#     def record_hi(self,UUT,hi):
#         self.hi_dict[UUT] = hi

#     def summary_scalar(self):
#         kvs = {}
#         for key in list(self.scalar_dict.keys()):
#             kvs[key] = sum(self.scalar_dict[key]) / len(self.scalar_dict[key])
#             del self.scalar_dict[key]
#         return kvs

#     def summary_scalars(self):
#         kvs = {}
#         for main_tag in list(self.scalars_dict.keys()):
#             kvs[main_tag] = {}
#             for key in self.scalars_dict[main_tag].keys():
#                 kvs[main_tag][key] = sum(self.scalars_dict[main_tag][key]) / len(self.scalars_dict[main_tag][key])
#             del self.scalars_dict[main_tag]
#         return kvs
    
#     def summary_histogram(self):
#         kvs = {}
#         for key in list(self.histogram_dict.keys()):
#             kvs[key] = sum(self.histogram_dict[key]) / len(self.histogram_dict[key])
#             del self.histogram_dict[key]
#         return kvs

#     def summary_hi(self):
#         kvs = self.hi_dict
#         self.hi_dict = {}
#         return kvs



class Logger:
    def __init__(self, args, log_name = None, comment = None, **writer_kwargs):
        if not log_name:
            log_name = args.model_name + '_' + args.time_str
        if comment:
            log_name = log_name + '_' + comment
        self.log_path = os.path.join(args.log_path, log_name)
        self.writer = SummaryWriter(self.log_path, **writer_kwargs)
        # self.recorder = Recorder()
        self.scalar_dict = defaultdict(list)
        self.scalars_dict = defaultdict(lambda: defaultdict(list))
        self.histogram_dict = defaultdict(list)
        self.normaltest_dict = defaultdict(list)
        # self.result_dir = args.result_dir
        self.checkpoint_path = args.checkpoint_path

    def record_scalar(self, tag, value):
        self.scalar_dict[tag].append(value)
        # self.recorder.record_scalar(tag, value)
    
    def record_scalars(self, main_tag, tag, value):
        self.scalars_dict[main_tag][tag].append(value)
        # self.recorder.record_scalars(main_tag, tag, value)

    def record_histogram(self, tag, values):
        self.histogram_dict[tag].append(values)
        # self.recorder.record_histogram(tag, values)

    def record_normaltest(self,test_name,test_result):
        self.normaltest_dict[test_name].extend(test_result)

    def save_scalar(self, global_step,):
        for tag in list(self.scalar_dict.keys()):
            scaler = sum(self.scalar_dict[tag]) / len(self.scalar_dict[tag])
            del self.scalar_dict[tag]
            self.writer.add_scalar(tag, scaler, global_step)

    def save_scalars(self, global_step):
        for main_tag in list(self.scalars_dict.keys()):
            tag_scalar_dict = {}    
            for tag in self.scalars_dict[main_tag].keys():
                tag_scalar_dict[tag] = sum(self.scalars_dict[main_tag][tag]) / len(self.scalars_dict[main_tag][tag])
            del self.scalars_dict[main_tag]
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def save_histogram(self, global_step,):
        for tag in list(self.histogram_dict.keys()):
            values = sum(self.histogram_dict[tag]) / len(self.histogram_dict[tag])
            del self.histogram_dict[tag]
            self.writer.add_histogram(tag, values, global_step)

    def save_metrics(self,global_step,):
        self.save_scalar(global_step,)
        self.save_scalars(global_step,)
        self.save_histogram(global_step,)

    def save_checkpoint(self, model, epoch, step=0):
        checkpoint_name = f'{epoch:03d}_{step:05d}.pth'
        model_fp = os.path.join(self.checkpoint_path, checkpoint_name)
        torch.save(model.state_dict(), model_fp)

    def __exit__(self):
        self.writer.close()
