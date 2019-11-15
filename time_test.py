
import time

def function(x):
    out = 1
    for i in range(x):
        out = out * i
        if i%1000 == 0:
            end_time = time.time()
            time_used = end_time-start_time
            print(time_used)
            print("time:{:.3}".format(time_used))
            m, s = divmod(time_used, 60)
            h, m = divmod(m, 60)
            print("%02d:%02d:%02.6f" % (h, m, s))
            print("{:2}:{:2}:{:2.6}".format(int(h),int(m),s))
    return out

print("tatal results as follows")
start_time = time.time()
out = function(10000000000000)
end_time = time.time()
time_used = end_time-start_time
print (time_used)

def changetoHSM(secends):
    m, s = divmod(time_used, 60)
    h, m = divmod(m, 60)
    return h,m,s
print ("%02d:%02d:%06d" % (h, m, s))
print("time used equal %f CPU seconds" % (time_used))

