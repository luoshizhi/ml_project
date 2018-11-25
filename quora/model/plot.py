import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
import pandas as pd
import sys
# import os


def plot(file, collist=[], title=None, savefig=None):
    data = pd.read_table(file)
    if not collist:
        collist = data.columns
    for key in collist:
        plt.plot(data.loc[:, key], label=key)
    plt.legend(loc='best')
    if title:
        plt.title(title)
    if not savefig:
        savefig = file+".jpg"
    plt.savefig(savefig)
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ("python {0} stat.txt".format(sys.argv[0]))
        sys.exit()

    plot(sys.argv[1],
         collist=["loss", "train_accuracy", "valid_accuracy"],
         title=sys.argv[2])
