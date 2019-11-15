#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWARE:PyCharm
@FILE:joselynstool.py
@TIME:2019/9/17 下午3:31
@DES:
'''


import sys
import  os
file_list = []
dirs = os.listdir("reid")
for dir in dirs :
    if dir[-3:]==".py" and  dir!= "__init__.py":
        file_list.append(dir)
    elif dir!="__init__.py" and dir!="__pycache__":
        # 是文件夹的情况
        files = os.listdir("reid/"+dir)
        for file in files:
            if file!="__init__.py" and file[-3:]==".py":
                file_path = dir+'/'+file
                file_list.append(file_path)
                # print(dir+" : "+file)
for file in file_list:
    print(file)

relations = []
for i in range(len(file_list)):
    rr = []
    for j in range(len(file_list)):
        rr.append(0)
    relations.append(rr)

for one in relations:
    print(one)
for i in range(len(file_list)):
    file = file_list[i]
    file_content = open("reid/"+file,'r')
    for j in range(len(file_list)):
        ff = file_list[j]

        if '/' in ff:
            ff=ff.split('/')[1]
        print(ff + " vs " + file),
        if ff[-3:] in file_content:
            relations[i][j] = 1
            print(relations[i][j])
for one in relations:
    print(one)


    # print(dir[-3:])


if __name__ == "__main__":
    print("hello joselyn!")