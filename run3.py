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
import time

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



class gif_drawer2():
    def __init__(self):
        plt.ion()
        self.select_num_percent = [0, 0]
        self.top1 = [0, 0]
        # self.select_num_percent =[0,0]
        self.mAP = [0,0]
        self.label_pre = [0,0]
        self.select_pre = [0,1]
        self.flag = 0
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
        plt.ylabel("value(%)"
                   )
        plt.plot(self.select_num_percent, self.top1, c="r", marker ='o',label="top1")
        # plt.plot(self.xs, self.select_num_percent, c="g", marker ='o',label="select_num_percent")
        plt.plot(self.select_num_percent, self.mAP, c="y", marker ='o',label="mAP")
        plt.plot(self.select_num_percent, self.label_pre, c="b", marker ='o',label="label_pre")
        plt.plot(self.select_num_percent, self.select_pre, c="cyan", marker ='o',label="select_pre")
        if self.flag==0:
            plt.legend()
            self.flag=1
        plt.pause(0.1)

    def saveimage(self,picture_path):
        plt.savefig(picture_path)

def changetoHSM(secends):
    m, s = divmod(secends, 60)
    h, m = divmod(m, 60)
    return h,m,s

import  codecs
def main(args):
    gd = gif_drawer2()

    print("game begin!")
    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.logs_dir
    total_step = math.ceil(math.pow((100 / args.EF), (1 / args.q))) + 1  # 这里应该取上限或者 +2  多一轮进行one-shot训练的
    print("total_step:{}".format(total_step))
    sys.stdout = Logger(osp.join(args.logs_dir, 'log' + str(args.EF)+"_"+ str(args.q) + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    data_file =codecs.open(osp.join(args.logs_dir,'data' + str(args.EF)+"_"+ str(args.q) + time.strftime(".%m_%d_%H-%M-%S") + '.txt'),'a')
    time_file =codecs.open(osp.join(args.logs_dir,'time' + str(args.EF)+"_"+ str(args.q) + time.strftime(".%m_%d_%H-%M-%S") + '.txt'),'a')

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
    # data_file = open("")
    start_time = time.clock()
    while(not isout):
        onetimeS = time.clock()
        print("This is running {} with EF ={}%, q = {} step {}:\t Nums_been_selected {}, \t Logs-dir {}".format(
            args.mode, args.EF, args.q, step, nums_to_select, save_path))
        onetime_trainS = time.clock()
        eug.train(new_train_data, step, epochs=70, step_size=55, init_lr=0.1) if step != resume_step else eug.resume(
            ckpt_file, step)
        onetime_trainE = time.clock()
        onetime_train = onetime_trainE-onetime_trainS
        h,m,s = changetoHSM(onetime_train)
        print("joselyn msg: traning is over,cost %02d:%02d:%02.6f" % (h, m, s))
        # evluate
        onetime_evaluateS = time.clock()
        mAP,top1,top5,top10,top20 = eug.evaluate(dataset_all.query, dataset_all.gallery)
        onetime_evaluateE = time.clock()
        onetime_evaluate = onetime_evaluateE-onetime_evaluateS
        h, m, s = changetoHSM(onetime_evaluate)
        step_size.append(nums_to_select)
        if nums_to_select==len(u_data):
            isout=1
        print("joselyn msg: evaluate is over,cost %02d:%02d:%02.6f" % (h, m, s))

        # pseudo-label and confidence sc
        nums_to_select = min(math.ceil(len(u_data) * math.pow((step + 1), args.q) * args.EF / 100),
                             len(u_data))  # 指数渐进策略
        onetime_estimateS = time.clock()
        pred_y, pred_score,label_pre,id_num= eug.estimate_label()
        onetime_estimateE = time.clock()
        onetime_estimate = onetime_estimateE-onetime_estimateS
        h, m, s = changetoHSM(onetime_estimate)
        print("joselyn msg: estimate labels is over,cost %02d:%02d:%02.6f" % (h, m, s))
        # select data
        selected_idx = eug.select_top_data(pred_score, nums_to_select)
        # selected_idx = eug.select_top_data(pred_score, nums_to_select,id_num,pred_y,u_data) #for 同比
        print("joselyn msg: select top data is over")
        # add new data
        new_train_data,select_pre = eug.generate_new_train_data(selected_idx,pred_y)
        # new_train_data,select_pre = eug.generate_new_train_data(selected_idx, pred_y) #for 同比
        print("joselyn msg: generate new train data is over")

        gd.draw(step_size[step]/len(u_data),top1,mAP,label_pre,select_pre)
        onetimeE =time.clock()
        onetime = onetimeE-onetimeS
        h, m, s = changetoHSM(onetime)
        data_file.write("step:{} top1:{:.2%} nums_selected:{} selected_percent:{:.2%} mAP:{:.2%} label_pre:{:.2%} select_pre:{:.2%}\n".format(int(step),top1,step_size[step],step_size[step]/len(u_data),mAP,label_pre,select_pre))
        time_file.write("step:{} traning:{:.8} evaluate:{:.8} estimate:{:.8} onetime:{:.8}\n".format(int(step),onetime_train,onetime_evaluate,onetime_estimate,onetime))
        print("step:{} top1:{:.2%} nums_selected:{} selected_percent:{:.2%} mAP:{:.2%} label_pre:{:.2%} select_pre:{:.2%}".format(int(step),top1,step_size[step],step_size[step]/len(u_data),mAP,label_pre,select_pre))
        print("onetime cost %02d:%02d:%02.6f" % (h, m, s))
        step = step + 1

    data_file.close()
    time_file.close()
    end_time = time.clock()
    alltime = end_time-start_time
    h, m, s = changetoHSM(alltime)
    print("alltime cost %02d:%02d:%02.6f" % (h, m, s))

    # gd.saveimage(osp.join(args.logs_dir,'image' + str(args.EF)+"_"+ str(args.q) + time.strftime(".%m_%d_%H-%M-%S")))











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
    parser.add_argument('--q', type=float, default=1)  # 指数参数
    parser.add_argument('--k', type=float, default=15)
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
