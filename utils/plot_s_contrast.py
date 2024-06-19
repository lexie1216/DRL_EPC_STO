# -*- ecoding: utf-8 -*-
# @ModuleName: plot_s_contrast
# @Function:
# @Time: 2024/4/24 19:45

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
import pickle
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
pd.set_option('display.max_columns', None)

class res_analysis_plot():
    def __init__(self,
                 city='tokyo',
                 R0='high',
                 model_idx=[100, 100, 100, 100],
                 experiment_idx=[-1, 1, 2, 3, 4, 5, 6]):
        self.city = city
        self.R0 = R0
        self.model_idx = model_idx

        self.street = geopandas.GeoDataFrame.from_file(f"../data/{self.city}/{self.city}.shp")
        self.street = self.street.sort_values(by='tid')
        # print(self.street.head(5))

        self.OD = np.load(f"../data/{self.city}/flow.npy")
        self.pop = np.load(f"../data/{self.city}/population.npy")
        self.ZONE_NUM = self.pop.shape[0]

        r, c = np.diag_indices_from(self.OD)
        self.OD[r, c] = 0
        self.inflow = self.pop.dot(self.OD)

        self.simRes = []
        self.actions = []

        for i in range(len(experiment_idx)):
            simRes, actions = self.load_data(experiment_idx[i], self.model_idx[i])
            self.simRes.append(simRes)
            self.actions.append(actions)
        # noNPI = np.load(f"../res/no_NPI_simRes_{self.city}_{self.R0}.npy")
        # self.simRes.append(noNPI)

        city_capacity = {
            'sz': 4e6,
            'tokyo': 2.2e6,
            'nyc': 2e6,
            'sh': 5.5e6
        }

        city_ylim = {
            'sz': 1.8e7,
            'tokyo': 1e7,
            'nyc': 1e7,
            'sh': 2.5e7
        }

        self.capacity = city_capacity[self.city]
        self.capacity = self.capacity if self.R0 == 'high' else self.capacity / 2

        self.ylim = city_ylim[self.city]



    def load_data(self, experiment_idx, model_idx):
        simRes = np.load(f"../res/model_{self.city}_{str(experiment_idx)}_{str(model_idx)}_{self.R0}_simRes.npy")
        actions = np.load(f"../res/model_{self.city}_{str(experiment_idx)}_{str(model_idx)}_{self.R0}_actions.npy")

        return simRes, actions

    def s_contrast(self, scene, sorder_slices):
        gs = gridspec.GridSpec(len(scene), len(sorder_slices))
        fig = plt.figure(figsize=(12, 7), dpi=300)
        ylabels = ['No-Order', 'S-Order-adj', 'S-Order-mob', 'S-Order-adm']

        # 创建 colormap 和 norm
        vmin = 0
        vmax = 2

        c = sns.color_palette("Blues")
        myColors = (c[0], c[2], c[4])
        cmap = LinearSegmentedColormap.from_list('Blues', myColors, len(myColors))

        bounds = [0, 1, 2, 3]
        norm = BoundaryNorm(bounds, len(bounds))

        for row in range(len(scene)):
            for col in range(len(sorder_slices)):

                day = sorder_slices[col]
                ax = fig.add_subplot(gs[row, col])
                # plt.colorbar(mappable=cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=bounds, boundaries=bounds,shrink=0.4)

                self.street['actions'] = self.actions[scene[row]][day]

                self.street.plot(ax=ax, column='actions',
                                 vmin=vmin, vmax=vmax,
                                 cmap=cmap, linewidth=0.2, edgecolor='lightgrey')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xticks([])  # 不显示x轴刻度线
                ax.set_yticks([])  # 不显示y轴刻度线
                ax.xaxis.set_label_coords(-0.1, 1.05)  # 调整 -0.1 和 1.05 来设置标签的位置
                ax.xaxis.set_label_position('bottom')

                flow = pd.read_csv(f"../data/{self.city}/flow.csv")


                action = self.actions[scene[row]][day]



                action_df = pd.DataFrame({'action': action})

                flow['action_x'] = flow['tid_x'].map(action_df['action'])
                flow['action_y'] = flow['tid_y'].map(action_df['action'])


                if row == 0:
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel("Day {}".format(day), fontsize=14)

                if col == 0:
                    ax.set_ylabel(ylabels[row], fontsize=14)

        # plt.suptitle("Spatial Distribution of Control Intensity",fontsize=16,y=0.05)
        # 绘制总的图例

        cax = fig.add_axes([0.85, 0.03, 0.1, 0.02])  # 调整 [0.15, 0.05, 0.7, 0.03] 来设置图例的位置和大小
        cb = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
        # cb.set_label("control intensity", loc='left')
        cb.ax.set_xticks([0.5, 1.5, 2.5])
        cb.ax.set_xticklabels([0, 1, 2])

        plt.tight_layout()
        plt.savefig("fig9.png")
        plt.show()



if __name__ == '__main__':
    myPlot = res_analysis_plot(city='sz', R0='high', model_idx=[80, 98, 99, 100],
                               experiment_idx=[4, 3, 5, -1])

    myPlot.s_contrast(scene=[0, 1,2,3], sorder_slices=[5,15,20,30,45])
    exit(0)

    myPlot = res_analysis_plot(city='tokyo', R0='high', model_idx=[100, 100, 100],
                               experiment_idx=[4, 3, 5])

    myPlot.s_contrast(scene=[0, 1,2], sorder_slices=[12, 18, 24])