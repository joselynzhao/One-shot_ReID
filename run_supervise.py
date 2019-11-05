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
        self.step = [0,0]
        self.top1 = [0, 0]
        # self.select_num_percent =[0,0]
        self.mAP = [0,0]

        self.flag = 0
        # plt.legend(loc="upper left")

    def draw(self, update_x, update_top1,mAP):

        self.top1[0] = self.top1[1]
        self.step[0] = self.step[1]
        # self.select_num_percent[0] = self.select_num_percent[1]
        self.mAP[0] = self.mAP[1]

        self.step[1] = update_x
        self.top1[1] = update_top1
        # self.select_num_percent[1] = select_num_percent
        self.mAP[1] = mAP


        plt.title("Performance monitoring")
        plt.xlabel("select_percent(%)")
        plt.ylabel("value(%)")
        plt.plot(self.step, self.top1, c="r", marker ='o',label="top1")
        # plt.plot(self.xs, self.select_num_percent, c="g", marker ='o',label="select_num_percent")
        plt.plot(self.step, self.mAP, c="y", marker ='o',label="mAP")
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

import os
import  codecs
def main(args):
    gd = gif_drawer2()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    # num_all_examples = len(dataset_all.train)
    l_data, u_data = get_one_shot_in_cam2(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))
    # num_all_examples = len(dataset_all.train)
    # l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
    #     dataset_all.name))
    resume_step, ckpt_file = -1, ''
    if args.resume:  # 重新训练的时候用
        resume_step, ckpt_file = resume(args)

        # initial the EUG algorithm
    eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
              data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=args.logs_dir,
              max_frames=args.max_frames)


    isout = 0  #用来标记是否应该结束训练
    start_time = time.time()
    for step in range(30):
        onetimeS = time.time()
        print("This is running {} with EF ={}%, q = {} step {}: \t Logs-dir {}".format(
            args.mode, args.EF, args.q, step+1, save_path))
        onetime_trainS = time.time()
        eug.train(l_data, step, epochs=20, step_size=55, init_lr=0.1) if step != resume_step else eug.resume(
            ckpt_file, step)
        onetime_trainE = time.time()
        onetime_train = onetime_trainE-onetime_trainS
        h,m,s = changetoHSM(onetime_train)
        print("joselyn msg: traning is over,cost %02d:%02d:%02.6f" % (h, m, s))
        # evluate
        onetime_evaluateS = time.time()
        mAP,top1,top5,top10,top20 = eug.evaluate(dataset_all.query, dataset_all.gallery)
        onetime_evaluateE = time.time()
        onetime_evaluate = onetime_evaluateE-onetime_evaluateS
        h, m, s = changetoHSM(onetime_evaluate)
        onetimeE = time.time()
        onetime = onetimeE - onetimeS
        gd.draw(step+1, top1, mAP)
        print("joselyn msg: evaluate is over,cost %02d:%02d:%02.6f" % (h, m, s))
        data_file.write("step:{} top1:{:.2%} mAP:{:.2%} \n".format(int(step),top1,mAP))
        time_file.write("step:{} traning:{:.8} evaluate:{:.8} onetime:{:.8}\n".format(int(step),onetime_train,onetime_evaluate,onetime))
        print("step:{} top1:{:.2%}  mAP:{:.2%}".format(int(step),top1,mAP))
        print("onetime cost %02d:%02d:%02.6f" % (h, m, s))
    data_file.close()
    time_file.close()
    end_time = time.time()
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
