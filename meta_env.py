# -*- ecoding: utf-8 -*-
# @ModuleName: meta_env
# @Function: 
# @Author: Lexie
# @Time: 2023/4/14 13:52
import random

import numpy as np
import gym
import matplotlib.pyplot as plt

ZN = 74


class EpidemicModel(gym.Env):
    def __init__(self, reward_mode=1):
        self.reward_mode = reward_mode

        # 子区域个数，完整模型74
        self.ZONE_NUM = 74

        # 子区域邻接关系
        self.ADJ = np.load('data/adj.npy', allow_pickle=True)
        self.ADJ = self.ADJ - 1

        # 子区域间流动
        self.OD = np.load("data/flow.npy")[:self.ZONE_NUM, :self.ZONE_NUM]

        # 将OD矩阵归一化到行之和为1
        self.OD = self.OD / self.OD.sum(axis=1, keepdims=1)

        # 子区域人口数量
        self.POP = np.load("data/population.npy")[:self.ZONE_NUM]

        # 不考虑区域间流动的OD矩阵
        # self.OD = np.identity(self.ZONE_NUM)

        # 传染病相关参数（感染者移动比例、beta、潜伏期、恢复期）
        self.Pm = 0.4
        self.betas = np.array([0.8] * self.ZONE_NUM)
        self.sigma = 1 / 3
        self.gamma = 1 / 7

        self.capacity = 4e6
        # 模拟周期
        self.period = 120

        # 状态的观测天数
        self.WINDOW_SIZE = 7

        self.reset()

        self.daily_new_E=[]

    def spatio_entropy(self, action):
        action = action.reshape(self.ZONE_NUM)
        spatial_ent = 0
        for i in range(self.ZONE_NUM):
            sample = action[self.ADJ[i].astype(int)]
            values, counts = np.unique(sample, return_counts=True)
            probabilities = counts / np.sum(counts)
            entropy = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 1 else 0
            spatial_ent += entropy
        return spatial_ent

    def reset(self):
        self.day = 1
        self.actions = [np.zeros(self.ZONE_NUM)]
        self.rewards = []

        self.simState = np.zeros((self.ZONE_NUM, 4))
        self.simState[:, 0] = self.POP

        self.set_init_seed()

        self.simRes = [self.simState]

        Is_window = np.pad(np.array(self.simRes)[:, :, 2], ((self.WINDOW_SIZE - self.day, 0), (0, 0)), 'constant',
                           constant_values=(0, 0))

        obs = Is_window.flatten()
        return obs

    def set_init_seed(self, init_infection=100):
        # 随机撒种子
        random.seed(3074)

        rand_list = [random.randint(0, self.ZONE_NUM - 1) for _ in range(init_infection)]
        rand_list = [i % self.ZONE_NUM for i in range(init_infection)]
        for sid in rand_list:
            self.simState[sid, 1] += 1
            self.simState[sid, 0] -= 1

    def step(self, action=np.zeros(ZN)):

        self.day += 1
        self.actions.append(action)

        # 仓室模型step

        contagious_infects = self.simState[:, 1] + self.Pm * self.simState[:, 2]

        contagious_toJ = np.sum(self.OD.T * contagious_infects, axis=1)
        contagious_toJ[contagious_toJ < 0.0] = 0.0

        all_toJ = np.sum(self.POP * self.OD.T, axis=1)
        contagious_ratio_toJ = contagious_toJ / all_toJ
        contagious_ratio_toJ[contagious_ratio_toJ < 0.0] = 0.0

        lam = np.sum(self.OD * contagious_ratio_toJ * self.betas * (1 - 0.25 * action), axis=1)
        lam[lam < 0.0] = 0.0

        dS = -self.simState[:, 0] * lam
        dE = self.simState[:, 0] * lam - self.sigma * self.simState[:, 1]
        dI = self.sigma * self.simState[:, 1] - self.gamma * self.simState[:, 2]
        dR = self.gamma * self.simState[:, 2]

        dState = [dS, dE, dI, dR]
        addE = self.simState[:, 0] * lam
        self.daily_new_E.append([addE.sum()])
        self.simState = self.simState + np.array(dState).T
        self.simRes.append(self.simState)

        # 构造state/obs

        if self.day >= self.WINDOW_SIZE:
            Is_window = np.array(self.simRes)[-self.WINDOW_SIZE:, :, 2]
        else:
            Is_window = np.pad(np.array(self.simRes)[:, :, 2], ((self.WINDOW_SIZE - self.day, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))

        obs = Is_window.flatten() / 1e4

        # 计算奖励
        total_infections = self.simState[:, 2].sum()

        health_cost = -max((total_infections - self.capacity) / self.capacity, 0)

        economy_cost = -action.sum() / self.ZONE_NUM
        total_recovery = self.simState[:, 3].sum()
        attack_rate = (total_recovery + total_infections) / 17493765
        if attack_rate > 0.5:
            economy_cost = economy_cost * 5

        t_order_cost = -np.abs(action - self.actions[-2]).sum() / self.ZONE_NUM
        s_order_cost = -self.spatio_entropy(action) / self.ZONE_NUM

        if self.reward_mode == 2:
            order_cost = t_order_cost * 2  # 时间秩序-2
        elif self.reward_mode == 3:
            order_cost = s_order_cost * 2  # 空间秩序-3
        elif self.reward_mode == 4:
            order_cost = 0  # 无秩序-4
        else:
            order_cost = t_order_cost + s_order_cost  # 时空秩序-1

        reward = health_cost * 60 + economy_cost + order_cost

        self.rewards.append(reward)

        info = {"day": self.day, "health": health_cost * 60, "economy": economy_cost, "order": order_cost * 2,
                "s_order": s_order_cost, "t_order": t_order_cost}
        done = False
        if self.day > self.period:
            done = True

        return obs, reward, done, info

    def render(self, title=''):
        self.simRes = np.array(self.simRes)
        self.actions = np.array(self.actions)
        plt.figure(dpi=120, figsize=(4.2, 3.6))
        plt.grid(linestyle='-.', axis='both')

        plt.plot(np.sum(self.simRes[:, :, 2], axis=1), label='Infections', color='orange', linewidth=2)

        plt.axhline(self.capacity, ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, 1e7)

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()
        plt.plot(np.mean(self.actions, axis=1), label='NPI', color='green', linewidth=2)

        plt.plot(self.actions, alpha=0.3, color='0.5')

        plt.ylim(-1, 3)
        plt.yticks(range(0, 3, 1))

        plt.title("Daily Current Infection " + title)

        plt.legend(fontsize=8, loc='upper right')

        plt.show()

        # for i in range(self.ZONE_NUM):
        #     plt.plot(self.actions.T[i], alpha=0.3, color='0.5')
        #     plt.ylim(-1, 4)
        #     plt.yticks(range(0, 4, 1))
        #     plt.show()

        return


if __name__ == '__main__':

    # actions = np.zeros((180, ZN))
    actions = np.ones((180, ZN))
    # actions = np.random.randint(3, size=(180, ZN))

    env = EpidemicModel()
    # actions[13:50, :] = 2

    for i in range(1):
        env.reset()
        ep_s = 0
        ep_r = 0
        while True:

            s_, r, done, info = env.step(action=actions[ep_s] * i)
            # print(s_.shape)
            ep_s += 1
            ep_r += r

            # print(ep_s,r)

            if done:
                env.render()
                np.save('no_intervention_simRe.npy', np.array(env.simRes))

                print("end of episode %d||  reward:%.2f, steps:%d" % (i, ep_r, ep_s))
                max_I = np.max(np.sum(env.simRes[:, :, 2], axis=1))
                overload = (max_I - env.capacity) / env.capacity
                print(overload)
                print(env.daily_new_E)

                import csv
                with open("daily_new_E.csv", 'w', newline='') as t:  # numline是来控制空的行数的
                    writer = csv.writer(t)  # 这一步是创建一个csv的写入器（个人理解）
                    writer.writerows(env.daily_new_E)  # 写入样本数据

                env.reset()
                break
