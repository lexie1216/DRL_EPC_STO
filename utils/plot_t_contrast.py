# -*- ecoding: utf-8 -*-
# @ModuleName: plot_t_contrast
# @Function: 
# @Author: Lexie
# @Time: 2024/3/4 21:03

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
pd.set_option('display.max_columns', None)

config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",}
plt.rcParams.update(config)
city = 'sz'

street = geopandas.GeoDataFrame.from_file(f"../data/{city}/{city}.shp")
street = street.sort_values(by='tid')

df1 = pd.DataFrame(street[['tid', 'name']])

df2 = pd.read_csv("../data/_raw_data/街道对照表.csv", encoding="GBK", header=None)
df2.columns = ['id', 'en', 'ch', 'district']

df = pd.merge(df1, df2, left_on='name', right_on='ch')
df = df[['tid', 'name', 'district']]

print(df)

model_idx = [80, 99]
experiment_idx = [4, 2]

R0 = 'high'

simResArr = []
actionArr = []

for i in range(len(model_idx)):
    simResArr.append(np.load(f"../res/model_{city}_{str(experiment_idx[i])}_{str(model_idx[i])}_{R0}_simRes.npy"))
    actionArr.append(np.load(f"../res/model_{city}_{str(experiment_idx[i])}_{str(model_idx[i])}_{R0}_actions.npy"))

c = sns.color_palette("Blues")
myColors = (c[0], c[2], c[4])
cmap = LinearSegmentedColormap.from_list('Blues', myColors, len(myColors))
bounds = [0, 1, 2, 3]
norm = BoundaryNorm(bounds, len(bounds))

start = 0
end = 50
districts = df['district'].unique()
fig, axs = plt.subplots(2, 1, figsize=(6, 4.8))
for dname in districts:

    for row in range(2):
        actions = actionArr[row][start:end + 1].T
        ax = axs[row]
        jds = df[df['district'] == dname]
        jid = jds['tid']
        action = actions[jid]
        ylabels = jds['name']

        ax = sns.heatmap(action, ax=ax, linewidths=0.003, cmap=cmap, yticklabels=False,
                         xticklabels=False, cbar=False)
        if row == 0:
            ax.set_ylabel("No-Order")
            ax.set_yticks(ticks=np.arange(0.5, len(ylabels) + 0.5),
                          labels=[ylabels.get(tid, '') for tid in ylabels.keys()], rotation=0)
        else:
            ax.set_ylabel("T-Order")
            ax.set_xlabel("Days")
            ax.set_xticks(np.arange(0.5, end - start + 1.5, 5), labels=range(start, end + 1, 5), rotation=0)
            ax.set_yticks(ticks=np.arange(0.5, len(ylabels) + 0.5),
                          labels=[ylabels.get(tid, '') for tid in ylabels.keys()], rotation=0)

    cax = fig.add_axes([0.85, 0.05, 0.1, 0.02])  # 调整 [0.15, 0.05, 0.7, 0.03] 来设置图例的位置和大小
    cb = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.ax.set_xticks([0.5, 1.5, 2.5])
    cb.ax.set_xticklabels([0, 1, 2])
    plt.tight_layout()
    plt.savefig(f"fig{dname}.png")
    # plt.show()


# jiedao = ["南澳", "园岭", "光明", "翠竹", "东门", "黄贝", "清水河", "笋岗"]
# jiedao = ["南澳", "光明", "翠竹", "东门", "莲塘", "沙头角"]
jiedao = ["布吉", "观湖", "南湖", "西丽", "沙河", "马峦","华强北"]

label = ['(Buji)','(Guanhu)','(Nanhu)','(Xili)','(Shahe)','(Maluan)','(Huaqiangbei)']
d = {}
for i in range(len(jiedao)):
    d[jiedao[i]]=label[i]

jds = df[df['name'].isin(jiedao)]
jid = jds['tid']
ylabels = jds['name']


fig, axs = plt.subplots(len(jiedao), 1, figsize=(10, 4), dpi=120)

for i in range(len(jiedao)):

    actions1 = actionArr[0][start:end + 1].T[jid.iloc[i]]
    actions2 = actionArr[1][start:end + 1].T[jid.iloc[i]]

    ax = axs[i]

    action = np.array([actions1, actions2])

    ax = sns.heatmap(action, ax=ax, linewidths=0.003, cmap=cmap, yticklabels=False,
                     xticklabels=False, cbar=False)

    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel(ylabels.iloc[i]+"街道", rotation=0, labelpad=24)
    # ax.set_ylabel(d[ylabels.iloc[i]], rotation=0, labelpad=28)

    ax.set_yticks(ticks=np.arange(0.5, 2.5),
                  labels=['Basic', 'T-Order'], rotation=0)
    if i == len(jiedao) - 1:
        ax.set_xlabel("Days")
        ax.set_xticks(np.arange(0.5, end - start + 1.5, 5), labels=range(start, end + 1, 5), rotation=0)

# cax = fig.add_axes([0.85, 0.05, 0.1, 0.02])  # 调整 [0.15, 0.05, 0.7, 0.03] 来设置图例的位置和大小
# cb = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
# cb.ax.set_xticks([0.5, 1.5, 2.5])
# cb.ax.set_xticklabels([0, 1, 2])

plt.tight_layout()
plt.show()
#
# for row in range(2):
#     actions = actionArr[row][start:end + 1].T
#     ax = axs[row]
#     jds = df[df['name'].isin(jiedao)]
#     jid = jds['tid']
#     action = actions[jid]
#     ylabels = jds['name']
#     print(action.shape)
#
#     ax = sns.heatmap(action, ax=ax, linewidths=0.003, cmap=cmap, yticklabels=False,
#                      xticklabels=False, cbar=False)
#     if row == 0:
#         ax.set_ylabel("No-Order")
#         ax.set_yticks(ticks=np.arange(0.5, len(ylabels) + 0.5),
#                       labels=[ylabels.get(tid, '') for tid in ylabels.keys()], rotation=0)
#     else:
#         ax.set_ylabel("T-Order")
#         ax.set_xlabel("Days")
#         ax.set_xticks(np.arange(0.5, end - start + 1.5, 5), labels=range(start, end + 1, 5), rotation=0)
#         ax.set_yticks(ticks=np.arange(0.5, len(ylabels) + 0.5),
#                       labels=[ylabels.get(tid, '') for tid in ylabels.keys()], rotation=0)
#
# cax = fig.add_axes([0.85, 0.05, 0.1, 0.02])  # 调整 [0.15, 0.05, 0.7, 0.03] 来设置图例的位置和大小
# cb = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
# cb.ax.set_xticks([0.5, 1.5, 2.5])
# cb.ax.set_xticklabels([0, 1, 2])
# plt.tight_layout()
# plt.show()
