# -*- ecoding: utf-8 -*-
# @ModuleName: mobility_validation
# @Function: 对比深圳市真实flow和用gravity/radiation模型生成的flow，选择最适合的模型
# @Author: Lexie
# @Time: 2023/12/27 19:40

#TODO：相似度还没算！

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from skmob import FlowDataFrame
from skmob.models.gravity import Gravity
from skmob.models.radiation import Radiation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def my_r2(X, y):
    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(X)

    r2 = r2_score(y, predicted)

    return r2


def scatter_plot(x, y):
    plt.figure(figsize=(6, 4.8), dpi=100)
    plt.scatter(x, y, color='blue')
    plt.xlabel('Log Truth')
    plt.ylabel('Log Predict')
    plt.title('Log Truth vs Log Predict Scatter Plot')
    plt.grid(True)
    plt.show()


def compare_flow(synth_fdf):
    generated_sz = pd.DataFrame(synth_fdf)
    generated_sz['flow'] = generated_sz['flow'] * 0.65

    flow_compare_sz = pd.merge(sz_flow_truth, generated_sz, on=['origin', 'destination'], how='inner')
    flow_compare_sz = flow_compare_sz[flow_compare_sz['flow_x'] > 0]
    flow_compare_sz.columns = [['origin', 'destination', 'truth', 'predict']]

    flow_compare_sz['truth_log'] = np.log(flow_compare_sz['truth'])
    flow_compare_sz['predict_log'] = np.log(flow_compare_sz['predict'])

    r2_no_log = my_r2(flow_compare_sz['truth'], flow_compare_sz['predict'])
    r2_log = my_r2(flow_compare_sz['truth_log'], flow_compare_sz['predict_log'])

    rmse = np.sqrt(mean_squared_error(flow_compare_sz['truth'], flow_compare_sz['predict']))
    mae = mean_absolute_error(flow_compare_sz['truth'], flow_compare_sz['predict'])
    mape = mean_absolute_percentage_error(flow_compare_sz['truth'], flow_compare_sz['predict'])

    print(f'R² Score: {r2_no_log:.4f}')
    print(f'R² Score after log: {r2_log:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.2f}%')

    scatter_plot(flow_compare_sz['truth_log'], flow_compare_sz['predict_log'])


def fit_generate(model):
    model.fit(sz_fdf, relevance_column='pop_truth')
    synth_fdf = model.generate(sz,
                               tile_id_column='tid',
                               tot_outflows_column='tot_outflow',
                               relevance_column='pop_truth',
                               out_format='probabilities')
    print(model)
    compare_flow(synth_fdf)


np.random.seed(0)

sz_flow = pd.read_csv("shenzhen_flow/flow_hb.csv")
sz_pop = pd.read_csv("shenzhen_pop/sz_pop.csv")
sz = gpd.read_file('shenzhen_shp/street_GCJ02.shp')

tot_outflows = sz_flow[sz_flow['id_x'] != sz_flow['id_y']].groupby(
    ['name_x', 'id_x']).agg({'flow_hb': 'sum'}).reset_index()

sz = sz.merge(sz_pop, left_on='JDNAME', right_on='name')
sz = sz.merge(tot_outflows, left_on='JDNAME', right_on='name_x')

sz = sz[['id_x', 'JDNAME', 'geometry', 'truth', 'predict', 'flow_hb']]
sz.columns = ['tid', 'name', 'geometry', 'pop_truth', 'pop_predict', 'tot_outflow']

sz['pop_truth'] = sz['pop_truth'].astype(int)
sz['pop_predict'] = sz['pop_predict'].astype(int)
sz.to_file('../sz/sz.shp', index=False,encoding='utf-8')
exit(0)

# 从sz_mobility_matrix得到sz_flow_truth，这是之后用于对比的df
sz_mobility_matrix = np.load("../sz/flow.npy")

rows, cols = sz_mobility_matrix.shape
origins, destinations = np.indices((rows, cols))
origins = origins.ravel().astype(str)
destinations = destinations.ravel().astype(str)

mobility = sz_mobility_matrix.ravel()
data = {
    'origin': origins,
    'destination': destinations,
    'flow': mobility
}

sz_flow_truth = pd.DataFrame(data)
sz_flow_truth = sz_flow_truth[sz_flow_truth['origin'] != sz_flow_truth['destination']]

# 从sz_flow到sz_fdf，这是用于拟合的fdf
flow_data = sz_flow
flow_data = flow_data[['id_x', 'id_y', 'flow_hb']]
flow_data.columns = ['origin', 'destination', 'flow']

sz_fdf = FlowDataFrame(flow_data, tile_id='tid', tessellation=sz, flow='flow')

model = Gravity(deterrence_func_type='exponential', gravity_type='singly constrained')
fit_generate(model)

model = Gravity(deterrence_func_type='power_law', gravity_type='singly constrained')
fit_generate(model)

model = Gravity(deterrence_func_type='exponential', gravity_type='globally constrained')
fit_generate(model)

model = Gravity(deterrence_func_type='power_law', gravity_type='globally constrained')
fit_generate(model)

model = Radiation()
rad_flows = model.generate(sz, tile_id_column='tid', tot_outflows_column='tot_outflow', relevance_column='pop_truth',
                           out_format='probabilities')
compare_flow(rad_flows)

