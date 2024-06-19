# -*- ecoding: utf-8 -*-
# @ModuleName: mobility_generation
# @Function: 用深圳市的home-based flow拟合gravity model，为其他三个城市生成flow
#
# @Time: 2023/12/27 19:41
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import geopandas as gpd
import skmob
from skmob.utils import utils, constants
from skmob.models.gravity import Gravity
from skmob.models.radiation import Radiation
from skmob import FlowDataFrame
from shapely.ops import nearest_points
from shapely.ops import cascaded_union

import warnings

warnings.filterwarnings("ignore")
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
np.random.seed(0)

sz = gpd.read_file("../sz/sz.shp")
sz_flow = pd.read_csv("shenzhen_flow/flow_hb.csv")

sz_flow = sz_flow[['id_x', 'id_y', 'flow_hb']]
sz_flow.columns = ['origin', 'destination', 'flow']

sz_fdf = FlowDataFrame(sz_flow, tile_id='tid', tessellation=sz, flow='flow')

model = Gravity(deterrence_func_type='power_law', gravity_type='singly constrained')
model.fit(sz_fdf, relevance_column='pop_truth')

print(model)


def gen_flow(city):
    city_flow = model.generate(city,
                               tile_id_column='name',
                               relevance_column='pop_truth',
                               out_format='probabilities')

    city_flow = city_flow.merge(city, left_on='origin', right_on='name', how='left')

    city_flow = city_flow.merge(city, left_on='destination', right_on='name', how='left')

    city_mobility = city_flow.pivot_table(index='tid_x', columns='tid_y', values='flow', fill_value=0).to_numpy()

    city_mobility = city_mobility * 0.65
    np.fill_diagonal(city_mobility, 0.35)
    # csv to plot
    gdf = city
    gdf['lon'] = gdf.geometry.centroid.x
    gdf['lat'] = gdf.geometry.centroid.y
    gdf = gdf[['name', 'lon', 'lat']]

    city_flow = pd.DataFrame(city_flow)  # FlowDataFrame->DataFrame
    city_flow = city_flow[['origin', 'destination', 'flow', 'pop_truth_x','tid_x','tid_y']]


    df = city_flow.merge(gdf, left_on='origin', right_on='name', how='left')
    df = df.merge(gdf, left_on='destination', right_on='name', how='left')

    df['flow'] = df['pop_truth_x'] * df['flow']

    df = df[['tid_x', 'tid_y', 'name_x', 'name_y', 'lon_x', 'lat_x', 'lon_y', 'lat_y', 'flow']]


    return city_mobility, df


def spatial_adj(city):
    city['geometry_buffer'] = city.buffer(0.001)  # 调整缓冲区大小以适应数据
    adjacency = gpd.sjoin(city, city, how='inner', op='intersects')
    adjacency = adjacency[adjacency['tid_right'] != adjacency['tid_left']]

    adjacency_list = adjacency.groupby('tid_right')['tid_left'].apply(list).reset_index()
    adj_dict = dict(zip(adjacency_list['tid_right'], adjacency_list['tid_left']))

    return adj_dict


def sz_flow_csv_process():
    city = 'sz'
    gdf = gpd.read_file(f"../{city}/{city}.shp")
    city_flow = pd.read_csv(f"./shenzhen_flow/flow_hb.csv")
    city_flow = city_flow[city_flow['name_x']!=city_flow['name_y']]

    gdf['lon'] = gdf.geometry.centroid.x
    gdf['lat'] = gdf.geometry.centroid.y
    gdf = gdf[['name', 'lon', 'lat']]

    city_flow = city_flow.merge(gdf, left_on='name_x', right_on='name', how='left')
    city_flow = city_flow[['name_x', 'id_x', 'name_y', 'id_y', 'flow_hb',
                           'lon', 'lat']]
    city_flow = city_flow.merge(gdf, left_on='name_y', right_on='name', how='left')
    city_flow = city_flow[['id_x', 'id_y', 'name_x', 'name_y', 'lon_x', 'lat_x', 'lon_y', 'lat_y', 'flow_hb']]
    city_flow.columns = ['tid_x', 'tid_y', 'name_x', 'name_y', 'lon_x', 'lat_x', 'lon_y', 'lat_y', 'flow']

    city_flow.to_csv(f"../{city}/flow.csv")

if __name__ == '__main__':

    sz_flow_csv_process()
    exit(0)

    cities = ['tokyo','nyc','sh']
    city = cities[0]

    city_shp = gpd.read_file(f"../{city}/{city}.shp")
    city_mobility,city_flow = gen_flow(city_shp)
    city_flow.to_csv(f"../{city}/flow.csv")
    np.save(f'../{city}/flow.npy', city_mobility)

    city_pop = np.array(city_shp['pop_truth'])
    np.save(f'../{city}/population.npy', city_pop)

    tokyo_adj_dict = spatial_adj(city_shp)
    with open('../tokyo/adj_dict.pkl', 'wb') as file:
        pickle.dump(tokyo_adj_dict, file)
