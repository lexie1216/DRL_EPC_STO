# -*- ecoding: utf-8 -*-
# @ModuleName: plot_seir
# @Function:
# @Time: 2024/3/3 21:59
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib import colorbar
import geopandas
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import rcParams

import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix", }
plt.rcParams.update(config)

simResArr = []

city = 'sz'
R0 = 'high'
ylim = 2e7
capacity = 4e6

for i in range(3):
    simRes = np.load(f"../res/simRes_{city}_{R0}_level_{i}.npy")
    simResArr.append(simRes)

simRes = np.load(f"../res/simRes_{city}_{R0}_dynamic.npy")
simResArr.append(simRes)
# simRes = np.load(f"../res/model_sz_4_80_high_simRes.npy")
# simRes = np.load(f"../res/model_sz_5_99_high_simRes.npy")
# simRes = np.load(f"../res/model_sz_6_100_high_simRes.npy")
simRes = np.load(f"../res/model_sz_-1_100_high_simRes.npy")
simResArr.append(simRes)


# simRes = np.load(f"../res/simRes_{city}_{R0}_level_0.npy")
# simResArr.append(simRes)
# simRes = np.load(f"../res/model_sz_1_100_high_simRes.npy")
# simResArr.append(simRes)

# 画2*2组图的，但是这个组图不太好用ppt加文字
def baselines(version='old'):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), dpi=200)
    lables = ['a', 'b', 'c', 'd']

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.grid(linestyle='-.', axis='both', alpha=0.5)
            index = i * 2 + j
            simRes = simResArr[index]

            ax.plot(np.sum(simRes[:, :, 0], axis=1), label='S', color='#C7C7C7', linewidth=2)
            ax.plot(np.sum(simRes[:, :, 1], axis=1), label='E', color='#FAC45A', linewidth=2)
            ax.plot(np.sum(simRes[:, :, 2], axis=1), label='I', color='#EC2835', linewidth=2)
            ax.plot(np.sum(simRes[:, :, 3], axis=1), label='R', color='#78A040', linewidth=2)

            ax.axhline(capacity, ls='-.', color='grey')
            ax.set_xlim(0, 120)
            ax.set_xticks(range(0, 121, 20))
            ax.set_ylim(0, ylim)
            ax.set_yticks(range(0, int(ylim), int(ylim / 5)))

            if version != 'new':
                ax.set_xlabel("Days", fontsize=14)
                ax.set_ylabel("Number of Individuals", fontsize=14)

            ax.legend(fontsize=12, loc='center right')

            if version != 'new':
                ax.text(0.9, 0.95, f'({lables[index]})', transform=ax.transAxes, fontsize=12, fontweight='bold',
                        va='top')

    plt.tight_layout()
    # plt.savefig("fig7.png")
    plt.show()


# baselines(version='new')

# 单独画 然后ppt组合
def baseline(index=0):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6), dpi=200)

    ax.grid(linestyle='-.', axis='both', alpha=0.5)

    simRes = simResArr[index]


    peak = np.max(np.sum(simRes[:, :, 2], axis=1))
    peak_day =  np.argmax(np.sum(simRes[:, :, 2], axis=1))
    print(peak,peak_day)


    ax.plot(np.sum(simRes[:, :, 0], axis=1), label='S', color='#C7C7C7', linewidth=2)
    ax.plot(np.sum(simRes[:, :, 1], axis=1), label='E', color='#FAC45A', linewidth=2)
    ax.plot(np.sum(simRes[:, :, 2], axis=1), label='I', color='#EC2835', linewidth=2)
    ax.plot(np.sum(simRes[:, :, 3], axis=1), label='R', color='#78A040', linewidth=2)

    ax.axhline(capacity, ls='--', color='grey',label='capacity')
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))
    ax.set_ylim(0, ylim)
    ax.set_yticks(range(0, int(ylim), int(ylim / 5)))

    # ax.legend(fontsize=10, loc='center right')
    ax.legend(fontsize=14)

    plt.tight_layout()
    # plt.savefig("fig7.png")
    plt.show()


def baseline_book(index=0):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6), dpi=200)

    ax.grid(linestyle='-.', axis='both', alpha=0.5)

    simRes = simResArr[index]

    ax.plot(np.sum(simRes[:, :, 0], axis=1), label='S', color='black', marker='s', linestyle='--', markersize=4)
    ax.plot(np.sum(simRes[:, :, 1], axis=1), label='E', color='black', marker='x', linestyle='--', markersize=4)
    ax.plot(np.sum(simRes[:, :, 2], axis=1), label='I', color='black', marker='^', linestyle='--', markersize=4)
    ax.plot(np.sum(simRes[:, :, 3], axis=1), label='R', color='black', marker='o', linestyle='--', markersize=4)

    ax.axhline(capacity, ls='-.', color='grey')
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))
    ax.set_ylim(0, ylim)
    ax.set_yticks(range(0, int(ylim), int(ylim / 5)))

    ax.legend(fontsize=12, loc='center right')

    plt.tight_layout()
    # plt.savefig("fig7.png")
    plt.show()


# for i in range(4):
#     baseline_book(i)


def baseline_distinct(index=0, region=1):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6), dpi=200)

    ax.grid(linestyle='-.', axis='both', alpha=0.5)

    simRes = simResArr[index]

    pop = np.max(simRes[:, region, 0])
    peak = np.max(simRes[:, region, 2])
    print(pop, peak, peak / pop)
    ylim = 3.5e5
    ylim = 1e5
    # print(pop)
    # if pop<3e5:
    #     print(pop,1)
    #     ylim=2.5e5
    # else:
    #     print(pop,2)
    #     ylim=4e5

    ax.plot(simRes[:, region, 0], label='S', color='#C7C7C7', linewidth=2)
    ax.plot(simRes[:, region, 1], label='E', color='#FAC45A', linewidth=2)
    ax.plot(simRes[:, region, 2], label='I', color='#EC2835', linewidth=2)
    ax.plot(simRes[:, region, 3], label='R', color='#78A040', linewidth=2)

    ax.axhline(capacity, ls='-.', color='grey')
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))
    ax.set_ylim(0, ylim)
    ax.set_yticks(range(0, int(ylim), int(2e4)))
    # ax.set_ylim(0, 2.5e5)
    # ax.set_yticks(range(0, int(2.5e5), int(2.5e5 / 5)))

    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    # plt.savefig("fig7.png")
    plt.show()


def action_distinct(region=1):
    action = np.load(f"../res/model_sz_1_100_high_actions.npy")

    action = np.load(f"../res/model_sz_5_99_high_actions.npy")
    action = np.load(f"../res/model_sz_6_100_high_actions.npy")
    action = np.load(f"../res/model_sz_-1_100_high_actions.npy")

    a = action[:, region]
    for i in range(len(a) - 1):
        if a[i] != a[i + 1]:
            print(i + 1, a[i], a[i + 1])

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6), dpi=200)

    ax.grid(linestyle='-.', axis='both', alpha=0.5)

    ax.plot(action[:, region], label='Control Intensity', color='royalblue', linewidth=2)

    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))

    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks(range(0, 3, 1))

    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()

    plt.show()


def plot_actions():
    actions = np.load(f"../res/model_sz_4_80_high_actions.npy")
    actions = np.load(f"../res/model_sz_5_99_high_actions.npy")
    actions = np.load(f"../res/model_sz_6_100_high_actions.npy")
    actions = np.load(f"../res/model_sz_-1_100_high_actions.npy")
    for i in range(120):
        print(i, np.mean(actions, axis=1)[i])

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6), dpi=200)
    ax.plot(np.mean(actions, axis=1), label='                 ', color='royalblue', linewidth=2, zorder=10)
    ax.plot(actions[0], label='                 ', alpha=0.2, color='0.5')

    # ax.plot(np.mean(actions, axis=1), label='                 ', color='royalblue', linewidth=2, zorder=10)
    # ax.plot(actions[0], label='                 ', alpha=0.2, color='0.5')
    ax.plot(actions[1:], alpha=0.2, color='0.5')

    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks(range(0, 3, 1))
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))
    # ax.set_xlabel("Days", fontsize=12)
    # ax.set_ylabel("Control Intensity", fontsize=12)
    # ax.legend(fontsize=12, loc='upper right')
    # ax.grid(linestyle='-.', axis='both', alpha=0.5)
    plt.show()


# # baseline(0)
# baseline_distinct(0,1)
# baseline_distinct(0,4)
#
# #drl
# baseline_distinct(1,1)
# baseline_distinct(1,4)


# action_distinct(0,4)

# for i in range(74):
#     action_distinct(0,i)


# action_distinct(0,12)
# action_distinct(0,16)
#
#
# baseline_distinct(1,12)
# baseline_distinct(0,12)
# baseline_distinct(1,16)
# baseline_distinct(0,16)


# action_distinct(0,20)
# action_distinct(0,21)
#
#
# baseline_distinct(1,20)
# baseline_distinct(0,20)
# baseline_distinct(1,21)
# baseline_distinct(0,21)

def infection_contrast(region):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6), dpi=200)

    ax.grid(linestyle='-.', axis='both', alpha=0.5)
    # zero control
    simRes = simResArr[0]

    pop = np.max(simRes[:, region, 0])
    peak = np.max(simRes[:, region, 2])
    print(pop, peak, peak / pop)
    ylim = 3.5e5
    ylim = 1e5

    ax.plot(simRes[:, region, 2], linestyle='-.', label='Infection Curve with Zero Control', color='#EC2835',
            linewidth=2, zorder=10)
    # mob
    simRes = simResArr[4]
    ax.plot(simRes[:, region, 2], color='#EC2835',  label = 'Infection Curve with DRL Control',linewidth=2, zorder=1)
    pop = np.max(simRes[:, region, 0])
    peak = np.max(simRes[:, region, 2])
    print(pop, peak, peak / pop)

    ax.axhline(capacity, ls='-.', color='grey')
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))
    ax.set_ylim(0, ylim)
    ax.set_yticks(range(0, int(ylim), int(2e4)))
    # ax.set_ylim(0, 2.5e5)
    # ax.set_yticks(range(0, int(2.5e5), int(2.5e5 / 5)))

    ax.legend(fontsize=12, loc='upper right')
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 4))  # 设置指数的上下界
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    # plt.savefig("fig7.png")
    plt.show()




def infection_wider():
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), dpi=200)

    ax.grid(linestyle='-.', axis='both', alpha=0.5)

    simRes = np.load(f"../res/model_sz_4_80_high_simRes.npy")

    ylim = 5e6

    infection_curve = np.sum(simRes[:, :, 2], axis=1)
    ax.plot(infection_curve, label='Infection curve with DRL-Basic control', color='#EC2835', linewidth=2)

    # 标出特定点
    points_x = [5, 15, 20, 30, 45]
    points_y = infection_curve[points_x]  # 提取对应x值处的y值
    ax.scatter(points_x, points_y, color='black', marker='o', s=20, zorder=5)  # 使用scatter绘制点
    ax.annotate(f'Day 5', (0, points_y[0]), textcoords='offset points', xytext=(5, 10), ha='left',fontsize=7)
    ax.annotate(f'Day 15', (8, points_y[1]), textcoords='offset points', xytext=(5, 10), ha='left',fontsize=7)
    ax.annotate(f'Day 20', (15, points_y[2]), textcoords='offset points', xytext=(2, 15), ha='left',fontsize=7)
    ax.annotate(f'Day 30', (30, points_y[3]), textcoords='offset points', xytext=(8, 10), ha='left', fontsize=7)
    ax.annotate(f'Day 45', (40, points_y[4]), textcoords='offset points', xytext=(2, 10), ha='left',fontsize=7)


    ax.axhline(capacity, ls='--', color='grey',label='Medical carrying capacity')
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))
    ax.set_ylim(0, ylim)
    ax.set_yticks(range(0, int(ylim), int(1e6)))

    ax.legend(fontsize=7, loc='center right')
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 4))  # 设置指数的上下界
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    # plt.savefig("fig7.png")
    plt.show()



def infection_wider_PPT():
    fig, ax = plt.subplots(1, 1, figsize=(4,3.6), dpi=200)

    ax.grid(linestyle='-.', axis='both', alpha=0.5)

    simRes = np.load(f"../res/model_sz_4_80_high_simRes.npy")

    ylim = 5e6

    infection_curve = np.sum(simRes[:, :, 2], axis=1)
    ax.plot(infection_curve, label='Infection curve with DRL-Basic control', color='#EC2835', linewidth=2)

    # 标出特定点
    points_x = [5, 15, 20, 30, 45]
    points_y = infection_curve[points_x]  # 提取对应x值处的y值
    ax.scatter(points_x, points_y, color='black', marker='o', s=20, zorder=5)  # 使用scatter绘制点
    ax.annotate(f'Day 5', (0, points_y[0]), textcoords='offset points', xytext=(5, 10), ha='left',fontsize=10)
    ax.annotate(f'Day 15', (8, points_y[1]), textcoords='offset points', xytext=(0, 20), ha='left',fontsize=10)
    ax.annotate(f'Day 20', (15, points_y[2]), textcoords='offset points', xytext=(20, 5), ha='left',fontsize=10)
    ax.annotate(f'Day 30', (30, points_y[3]), textcoords='offset points', xytext=(8, 10), ha='left', fontsize=10)
    ax.annotate(f'Day 45', (40, points_y[4]), textcoords='offset points', xytext=(2, 10), ha='left',fontsize=10)


    ax.axhline(capacity, ls='--', color='grey',label='Medical carrying capacity')
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 121, 20))
    ax.set_ylim(0, ylim)
    ax.set_yticks(range(0, int(ylim), int(1e6)))

    ax.legend(fontsize=7, loc='upper right')
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 4))  # 设置指数的上下界
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    # plt.savefig("fig7.png")
    plt.show()

if __name__ == '__main__':
    for i in range(5):
        baseline(i)


    plot_actions()

    action_distinct(10)
    infection_contrast(10)



    infection_wider()

