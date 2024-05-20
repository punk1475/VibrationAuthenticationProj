import os

import pandas as pd
import matplotlib.pyplot as plt

from myutil import MyUtil

if __name__ == "__main__":
    csv_pd = pd.read_csv(r"C:\Users\25807\Downloads/test0.csv")

    acc_x = csv_pd.loc[:130, "acc_x"].values
    plt.plot(acc_x)
    plt.show()
#     unknown media notif
