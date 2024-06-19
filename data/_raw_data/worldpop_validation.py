# -*- ecoding: utf-8 -*-
# @ModuleName: worldpop_validation
# @Function: 以深圳市为例，分析worldpop数据是否可以作为一种替代方案
# @Output: R2:0.7，输出一个clean版的csv，只有name，truth，predict
#  
# @Time: 2023/12/27 19:27

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 深圳市分社区的人口数据，将其聚合到街道级别
sz_pop_truth = pd.read_excel("shenzhen_pop/20220420七人普各村（居）分年龄人口数.xls")
sz_pop_truth= sz_pop_truth[:-2]

sz_pop_truth = sz_pop_truth[['街道', '总数']]
sz_pop_truth['街道'] = sz_pop_truth['街道'].apply(lambda x: x[:-2])
sz_pop_truth= sz_pop_truth.groupby('街道').agg({'总数': 'sum'}).reset_index()

# 深圳市分街道的worldpop数据，原始数据是100m的，在arcmap上做了聚合处理
sz_worldpop = pd.read_excel("shenzhen_pop/street_pop.xls")

sz_pop =pd.merge(sz_pop_truth, sz_worldpop , left_on='街道', right_on='JDNAME')

sz_pop = sz_pop[['JDNAME', '总数', 'SUM']]
sz_pop.columns = ['name', 'truth', 'predict']

sz_pop['truth_log'] = np.log(sz_pop['truth'])
sz_pop['predict_log'] = np.log(sz_pop['predict'])


# 计算log变换后的R2值
X = sz_pop['truth_log'].values.reshape(-1, 1)
y = sz_pop['predict_log'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
predicted = model.predict(X)

r_squared = r2_score(y, predicted)
print(f"R^2 score: {r_squared}")


# 绘制log变换后的散点图
plt.figure(figsize=(6, 4.8), dpi=100)
plt.scatter(sz_pop['truth_log'], sz_pop['predict_log'])
plt.title('Scatter Plot of log(Truth) vs log(Predict)')
plt.xlabel('log(Truth)')
plt.ylabel('log(Predict)')
plt.grid(ls='-.', lw=0.5, color='gray')
plt.show()

sz_pop[['name','truth','predict']].to_csv("shenzhen_pop/sz_pop.csv")