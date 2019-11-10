import numpy as np

import matplotlib.pyplot as plt


if __name__=="__main__":
    b = np.load("dists.npy")
    counts = np.zeros(10)
    for raw in b:
        for col in raw:
            if col >=0.9:
                counts[9]=counts[9]+1
            elif 0.8<=col <0.9:
                counts[8]=counts[8]+1
            elif 0.7<= col <0.8:
                counts[7]=counts[7]+1
            elif 0.6<= col <0.7:
                counts[6]=counts[6]+1
            elif 0.5<= col <0.6:
                counts[5]=counts[5]+1
            elif 0.4<= col <0.5:
                counts[4]=counts[4]+1
            elif 0.3<=col <0.4:
                counts[3]=counts[3]+1
            elif 0.2<= col <0.3:
                counts[2]=counts[2]+1
            elif 0.1<= col <0.2:
                counts[1]=counts[1]+1
            elif 0.0<= col <0.1:
                counts[0]=counts[0]+1

    print(counts)
    x = np.linspace(1,10,10)
    plt.plot(x,counts)
    plt.show()

    # m = np.load("masks.npy")
    # for row in b:
    #     for clo in row:

    # print(m)