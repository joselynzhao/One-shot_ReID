import  pickle

df = open("oneshot_mars_used_in_paper.pickle","rb")
data = pickle.load(df)

for one in data:
    print(one)