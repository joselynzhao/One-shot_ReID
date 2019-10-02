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

import matplotlib.pyplot as plt


def resume(args):
    import re
    pattern = re.compile(r'step_(\d+)\.ckpt')
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


class gif_drawer():
    def __inti__(self):
        plt.ion()
        self.xs = [0, 0]
        self.ys = [0, 0]

    def draw(self, update_x, update_y):
        self.xs[0] = self.xs[1]
        self.ys[0] = self.ys[1]

        self.xs[1] = update_x
        self.ys[1] = update_y

        plt.title("top1 by steps")
        plt.xlabel("steps")
        plt.ylabel("vl%")
        plt.plot(self.xs, self.ys, c='r')
        # plt.pause(0.1)


class gif_drawer2():
    def __init__(self):
        plt.ion()
        self.select_num_percent = [0, 0]
        self.top1 = [0, 0]
        # self.select_num_percent =[0,0]
        self.mAP = [0,0]
        self.label_pre = [0,0]
        self.select_pre = [0,1]
        # plt.legend(loc="upper left")

    def draw(self, update_x, update_top1,mAP,label_pre,select_pre):
        self.select_num_percent[0] = self.select_num_percent[1]
        self.top1[0] = self.top1[1]
        # self.select_num_percent[0] = self.select_num_percent[1]
        self.mAP[0] = self.mAP[1]
        self.label_pre[0] = self.label_pre[1]
        self.select_pre[0] = self.select_pre[1]

        self.select_num_percent[1] = update_x
        self.top1[1] = update_top1
        # self.select_num_percent[1] = select_num_percent
        self.mAP[1] = mAP
        self.label_pre[1] = label_pre
        self.select_pre[1] = select_pre

        plt.title("Performance monitoring")
        plt.xlabel("select_percent(%)")
        plt.ylabel("value(%)")
        plt.plot(self.select_num_percent, self.top1, c="r", marker ='o',label="top1")
        # plt.plot(self.xs, self.select_num_percent, c="g", marker ='o',label="select_num_percent")
        plt.plot(self.select_num_percent, self.mAP, c="y", marker ='o',label="mAP")
        plt.plot(self.select_num_percent, self.label_pre, c="b", marker ='o',label="label_pre")
        plt.plot(self.select_num_percent, self.select_pre, c="cyan", marker ='o',label="select_pre")
        plt.pause(0.1)


def main(args):
    gd = gif_drawer2()
    print("game begin!")
    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.logs_dir
    total_step = math.ceil(math.pow((100 / args.EF), (1 / args.q))) + 1  # 这里应该取上限或者 +2  多一轮进行one-shot训练的
    sys.stdout = Logger(osp.join(args.logs_dir, 'log' + str(args.bs) + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    num_all_examples = len(dataset_all.train)
    l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))

    resume_step, ckpt_file = -1, ''
    if args.resume:  # 重新训练的时候用
        resume_step, ckpt_file = resume(args)

        # initial the EUG algorithm
    eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=args.logs_dir,
              max_frames=args.max_frames)

    nums_to_select = 0
    new_train_data = l_data
    step = 0
    # to_list = []
    step_size = []
    base_step = args.bs
    top_list = []  # top1
    isout = 0  #用来标记是否应该结束训练
    while(not isout):
        print("This is running {} with base_size ={}%, step {}:\t Nums_been_selected {}, \t Logs-dir {}".format(
            args.mode, args.bs, step, nums_to_select, save_path))
        eug.train(new_train_data, step, epochs=20, step_size=55, init_lr=0.1) if step != resume_step else eug.resume(
            ckpt_file, step)
        print("joselyn msg: ------------------------------------------------------traning is over")
        # evluate
        mAP,top1,top5,top10,top20 = eug.evaluate(dataset_all.query, dataset_all.gallery)
        top_list.append(top1)
        step_size.append(nums_to_select)
        if nums_to_select==len(u_data):
            isout=1
        print("joselyn msg: ------------------------------------------------------evaluate is over")

        # pseudo-label and confidence sc
        if(step==0): #初始化k0
            global k0
            k0 = 1-top1
            print("joselyn msg: ------------k0 = {}".format(k0))
            nums_to_select = base_step

        else:
            k = (top1-top_list[step-1])/step_size[step]
            delta_k = max(k-k0,-1)  # 可以在这里尝试除以k0，k0是不会等于0的
            print("joselyn msg: ------------new k = {}  delta_k = {}".format(k,delta_k))
            nums_to_select = step_size[step]+ math.floor(base_step*(1+delta_k))+1 # 这个值必须是整数
            if nums_to_select>len(u_data):
                nums_to_select = len(u_data)

        pred_y, pred_score,label_pre = eug.estimate_label()
        print("joselyn msg: estimate labels is over")
        # select data
        selected_idx = eug.select_top_data(pred_score, nums_to_select)
        print("joselyn msg: select top data is over")
        # add new data
        new_train_data,select_pre = eug.generate_new_train_data(selected_idx, pred_y)
        print("joselyn msg: generate new train data is over")

        gd.draw(step_size[step]/len(u_data),top1,mAP,label_pre,select_pre)
        print("step:{} top1:{:.2%} nums_selected:{} selected_percent:{:.2%} mAP:{:.2%} label_pre:{:.2%} select_pre:{:.2%}".format(int(step),top1,step_size[step],step_size[step]/len(u_data),mAP,label_pre,select_pre))
        step = step + 1











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
    # parser.add_argument('--EF', type=float, default=5)
    # parser.add_argument('--q', type=float, default=1)  # 指数参数
    # parser.add_argument('--k', type=float, default=15)
    parser.add_argument('--bs', type=int, default=50)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--continuous', action="store_true")
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"], default="Dissimilarity")
    parser.add_argument('--max_frames', type=int, default=100)
    main(parser.parse_args())
