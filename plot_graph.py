import matplotlib.pyplot as plt
import os
import glob
import numpy as np


def main():
    data_list = glob.glob(os.path.join('./', '*.data'))
    for data_fn in data_list:
        x, loss_train, acc_test = np.loadtxt(data_fn, dtype=np.float32)
        x = x.astype(np.int32)

        # Two subplots
        title = os.path.splitext(os.path.basename(data_fn))[0]
        save_fn = '{:s}.png'.format(title)
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(x, loss_train, color='C0', label='Loss')
        axarr[0].legend(loc='upper right')
        axarr[1].plot(x, acc_test, color='C1', label='Accuracy')
        axarr[1].legend(loc='lower right')
        axarr[0].set_title('{:s} - LOSS, ACCURACY'.format(title))
        plt.savefig(save_fn)
    return


if __name__ == '__main__':
    main()
