import multiprocessing
import threading
import multiprocessing
import csv
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import geometry
from pyproj import CRS
from pyproj import Transformer

TIME_START = 20190101000000  # %Y%m
TIME_END = 20201231235959  # %Y%m
TIME_STEP = 30  # minutes

crs_WGS84 = CRS.from_epsg(4326)
crs_taxi_zones = CRS.from_epsg(2263)
transformer = Transformer.from_crs(crs_WGS84, crs_taxi_zones)
manhattan_zones_file = open(
    './data-NYCZones/zones/manhattan_zones.csv', encoding='utf-8')
manhattan_zones_reader = csv.reader(manhattan_zones_file)
taxi_zones = gpd.read_file("./data-NYCZones/zones/taxi_zones.shp")
manhattan_zones_numbers = []
manhattan_zones_header = []
manhattan_zones = []
for i in manhattan_zones_reader:
    if len(manhattan_zones_header) == 0:
        manhattan_zones_header = i
        continue
    manhattan_zones_numbers.append(i)

for zone in range(len(taxi_zones)):
    id = taxi_zones["LocationID"][zone]
    for i in manhattan_zones_numbers:
        if int(i[0]) == int(id):
            manhattan_zones.append(taxi_zones["geometry"][zone])
            break


def cal_time(time):
    date = int(time.split()[0].split('-')[2])
    hour = int(time.split()[1].split(':')[0])
    minutes = int(time.split()[1].split(':')[1])
    num = int(((((date - 1) * 24 + hour))*60 + minutes)/TIME_STEP)
    return num


def get_flow(filepath):
    print(filepath)
    tot_step = int(31 * 24 * 60/int(TIME_STEP))
    inout_flow = np.zeros((2, tot_step, 69))
    od_flow = np.zeros((tot_step, 69, 69))
    file = open(filepath, encoding='utf-8')
    reader = csv.DictReader(file)
    cnt = 0
    for t in reader:
        start_id = -1
        end_id = -1
        start_time = cal_time(t["starttime"])
        stop_time = cal_time(t["stoptime"])
        start_latitude = t["start station latitude"]
        start_longitude = t["start station longitude"]
        start_x, start_y = transformer.transform(
            start_latitude, start_longitude)
        for zone in range(len(manhattan_zones)):
            if geometry.Point(start_x, start_y).within(manhattan_zones[zone]):
                inout_flow[0][start_time][zone] += 1
                start_id = zone
                break
        end_latitude = t["end station latitude"]
        end_longitude = t["end station longitude"]
        end_x, end_y = transformer.transform(end_latitude, end_longitude)
        for zone in range(len(manhattan_zones)):
            if geometry.Point(end_x, end_y).within(manhattan_zones[zone]):
                inout_flow[1][stop_time][zone] += 1
                end_id = zone
                break
        if start_id != -1 and end_id != -1 and start_time == stop_time:
            od_flow[start_time][start_id][end_id] += 1
        cnt = cnt + 1
        if cnt % 100000 == 0:
            print(cnt)
    print(np.any(inout_flow == 0))
    print(np.any(od_flow == 0))
    with open('./data-NYCBike/{}min/{}-graph-inoutflow.npz'.format(TIME_STEP,
            filepath.split("/")[-1].split(".")[0]), 'wb') as f:
        np.savez_compressed(f, inout_flow)
    with open('./data-NYCBike/{}min/{}-graph-odflow.npz'.format(TIME_STEP,
            filepath.split("/")[-1].split(".")[0]), 'wb') as f:
        np.savez_compressed(f, od_flow)
    print(filepath.split("/")[-1].split(".")[0] + "complete!")


if __name__ == '__main__':
    filepath_list = []
    for i in range(1, 13):
        filepath_list.append(
            '../../../data/mhy/NYCBikeData/2019/2019{}-citibike-tripdata.csv'.format(str(i).zfill(2)))
        filepath_list.append(
            '../../../data/mhy/NYCBikeData/2020/2020{}-citibike-tripdata.csv'.format(str(i).zfill(2)))
    pool = multiprocessing.Pool(processes=24)
    for i in range(len(filepath_list)):
        pool.apply_async(get_flow, (filepath_list[i],))
    pool.close()
    pool.join()
    print("Sub-process all done.")
