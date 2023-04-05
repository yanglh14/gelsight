import matplotlib.pyplot as plt # plotting library
import os
import numpy as np
import cv2

def plot_loss():
    data1 = np.load('./model/loss.npy',allow_pickle=True)
    data1= data1.item()

    data2 = np.load('./model/loss2.npy',allow_pickle=True)
    data2= data2.item()

    data3 = np.load('./model/loss3.npy',allow_pickle=True)
    data3= data3.item()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6)
                            )

    axs[0].plot(data1['train_loss'],label='train_loss50')

    axs[1].plot(data1['val_loss'],label='val_loss50')

    axs[0].plot(data2['train_loss'],label='train_loss200')

    axs[1].plot(data2['val_loss'],label='val_loss200')

    axs[0].plot(data3['train_loss'],label='train_loss600')

    axs[1].plot(data3['val_loss'],label='val_loss600')

    plt.legend()
    plt.show()

def plot_data():

    # make data
    dataset = np.load('data/cal_dataset2.npy', allow_pickle=True)
    x1 = dataset[:,-2]
    x2 = dataset[:,-1]

    # plot:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].hist(x1, bins=100, linewidth=0.1, edgecolor="white",label='Gx')
    ax[1].hist(x2, bins=100, linewidth=0.1, edgecolor="white",label='Gy')

    # ax.set(xlim=(-5, 5), xticks=np.arange(-5, 5,0.5))
    ax[0].legend()
    ax[1].legend()

    plt.show()

def plot_cal():
    # make data
    dataset = np.load('data/cal_dataset2.npy', allow_pickle=True)
    x1 = dataset[:,-2]
    x2 = dataset[:,-1]

    mask = np.zeros((49, 49), dtype=np.uint8)
    mask = cv2.circle(np.array(mask), (24, 24), int(24), (255, 255, 255), -1)
    mask = mask/255
    R = 24.2
    x = np.linspace(-24, 24, 49)
    y = np.linspace(-24, 24, 49)
    x, y = np.meshgrid(x, y)
    xv = x - 0
    yv = y - 0
    gx = (-xv*mask) / np.sqrt(R ** 2 - (xv*mask) ** 2 - (yv*mask) ** 2)
    gy = (-yv*mask) / np.sqrt(R ** 2 - (xv*mask) ** 2 - (yv*mask) ** 2)

    gx = gx.flatten()
    gy = gy.flatten()
    gx = np.arctan(gx)
    gy = np.arctan(gy)
    x1 = np.arctan(x1)
    x2 = np.arctan(x2)

    # plot:
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    ax[0,0].hist(gx[gx!=0], bins=100, linewidth=0.1, edgecolor="red",label='Gx_ground')
    ax[0,1].hist(gy[gy!=0], bins=100, linewidth=0.1, edgecolor="red",label='Gy_ground')

    ax[1,0].hist(x1, bins=100, linewidth=0.1, edgecolor="green",label='Gx_dataset')
    ax[1,1].hist(x2, bins=100, linewidth=0.1, edgecolor="green",label='Gy_dataset')

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.show()

if __name__ == "__main__":

    plot_loss()