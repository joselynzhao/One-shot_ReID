from __future__ import print_function, absolute_import
from reid.eug import *
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os

from reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from torch import nn
import time
import math
import pickle


def resume(args):
    import re
    pattern=re.compile(r'step_(\d+)\.ckpt')
    start_step = -1
    ckpt_file = ""

    # find start step
    files = os.listdir(args.logs_dir)
    files.sort()
    for filename in files:
        try:
            iter_ = int(pattern.search(filename).groups()[0])
            if iter_ > start_step:
                start_step = iter_
                ckpt_file = osp.join(args.logs_dir, filename)
        except:
            continue

    # if need resume
    if start_step >= 0:
        print("continued from iter step", start_step)

    return start_step, ckpt_file





def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.logs_dir
    # total_step = 100//args.EF + 1   #这个公式为什么要这样计算
    total_step = math.ceil(math.pow((100/args.EF),(1/args.q))) + 1  #这里应该取上限或者 +2  多一轮进行one-shot训练的
    # total_step = math.ceil(math.pi/(args.k/100)) + 1   #这个是cos函数的
    print("joselyn msg: total_setp is {}".format(total_step))
    # sys.stdout = Logger(osp.join(args.logs_dir, 'log'+ str(args.EF)+ time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    sys.stdout = Logger(osp.join(args.logs_dir, 'log'+ str(args.k)+ time.strftime(".%m_%d_%H-%M-%S") + '.txt'))

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    num_all_examples = len(dataset_all.train)
    l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(dataset_all.name))
    
    resume_step, ckpt_file = -1, ''
    if args.resume:  #重新训练的时候用
        resume_step, ckpt_file = resume(args) 

    # initial the EUG algorithm 
    eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids, 
            data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=args.logs_dir, max_frames=args.max_frames)


    # nums_to_select = 0
    new_train_data = l_data
    # isout = 0  #用来标记是否应该结束训练
    for step in range(total_step):
        # for resume
        if step < resume_step: # resume 想表达的是接着训练的意思
            continue

        # print("This is running {} with EF={}%, q={}, step {}/{}:\t Nums_to_be_select {}, \t Logs-dir {}".format(
        #     args.mode, args.EF, args.q, step+1,total_step, nums_to_select, save_path))

        # nums_to_select = min(int(len(u_data) * (step + 1) * args.EF / 100), len(u_data))  # 感觉这里就是渐进策略
        nums_to_select = min(math.ceil(len(u_data) * math.pow((step+1),args.q)* args.EF / 100), len(u_data))  # 指数渐进策略

        # 尝试使用cos
        # nums_to_select = min(math.ceil(-(len(u_data) / 2) * np.cos((args.k / 100) * (step + 1)) + len(u_data) / 2),
        #                          len(u_data))
        if step == total_step - 1:  # 表示这是最后一轮
            nums_to_select = len(u_data)
            # 取最小值的原因是因为，最多也就是把所有未带标签的数据全部加进去
        print("This is running {} with k={}%, step {}/{}:\t Nums_to_be_select {}, \t Logs-dir {}".format(
                args.mode, args.k, step + 1, total_step, nums_to_select, save_path))


        # train the model or load ckpt
        eug.train(new_train_data, step, epochs=20, step_size=55, init_lr=0.1) if step != resume_step else eug.resume(ckpt_file, step)
        print("joselyn msg: traning is over")

        # evluate
        eug.evaluate(dataset_all.query, dataset_all.gallery)

        print("joselyn msg: evaluate is over")



        # pseudo-label and confidence sc
        pred_y, pred_score = eug.estimate_label()
        print("joselyn msg: estimate labels is over")
        # select data
        selected_idx = eug.select_top_data(pred_score, nums_to_select)
        print("joselyn msg: select top data is over")
        # add new data
        new_train_data = eug.generate_new_train_data(selected_idx, pred_y)
        print("joselyn msg: generate new train data is over")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exploit the Unknown Gradually')
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',
                        choices=models.names())
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--EF', type=float, default=5)
    parser.add_argument('--q', type=float, default=1)  #指数参数
    parser.add_argument('--k',type = float,default=15)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--continuous', action="store_true")
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"],default="Dissimilarity")
    parser.add_argument('--max_frames', type=int, default=100)
    main(parser.parse_args())
