<<<<<<< HEAD
import  pickle

df = open("oneshot_mars_used_in_paper.pickle","rb")
data = pickle.load(df)

for one in data:
    print(one)
=======
import  pickle

df = open("oneshot_mars_used_in_paper.pickle","rb")
# data = pickle.load(df)
#
# print(len(data))

dataset = pickle.load(df)
label_dataset = dataset["label set"]
unlabel_dataset = dataset["unlabel set"]

print(label_dataset)

df.close()


# #导入模块
# import pickle
#
# #准备要序列化的数据
# n = 7
# i = 13000000
# a = 99.056
# s = '中国人民 123abc'
# lst = [[1,2,3],[4,5,6],[7.8,9]]
# tu = (-5,10,8)
# coll = {4,5,6}
# dic = {'a':'apple','b':'banana','g':'grape','o':'orange'}
# aa = 'c'
#
# #以写模式打开二进制文件
# f = open('sample_pickle.pickle','wb')
# try:
#     pickle.dump(n,f)    #对象个数
#     pickle.dump(i,f)    #写入整数
#     pickle.dump(a,f)    #写入实数
#     pickle.dump(s,f)    #写入实数
#     pickle.dump(lst,f)  #写入列表
#     pickle.dump(tu,f)   #写入元组
#     pickle.dump(coll,f) #写入集合
#     pickle.dump(dic,f)  #写入字典
# except:
#     print('写文件异常')
# finally:
#     f.close()

#看一下文件内容，已经写入成功。
>>>>>>> ad932266200cccbfdc41d2993341b72323df8903
