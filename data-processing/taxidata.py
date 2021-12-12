import numpy as np
import pandas as pd
import multiprocessing

TIME_GAP = 0.5

def get_flow(year, month):
    in_outflow = np.zeros((2, int((31*24) / TIME_GAP), 69))
    odflow = np.zeros((int((31*24) / TIME_GAP), 69, 69))

    path = f'/home/cseadmin/data/mhy/NYCTaxiData/YellowTaxi{year}/yellow_tripdata_%d-%02d.csv' % (year, month)
    taxi_data = pd.read_csv(path, dtype=str)
    taxi_data['DOLocationID'] = taxi_data['DOLocationID'].astype('int64')
    taxi_data['PULocationID'] = taxi_data['PULocationID'].astype('int64')
    manhattan_zones = pd.read_csv("/home/cseadmin/mhy/data/NYCZones/manhattan_zones.csv")
    manhattan_zones_id  = list(manhattan_zones["zone_id"])
    manhattan_taxi_data = taxi_data[(taxi_data['DOLocationID'].isin(manhattan_zones_id)) & (taxi_data['PULocationID'].isin(manhattan_zones_id))]
    
    manhattan_taxi_data = manhattan_taxi_data[manhattan_taxi_data["tpep_pickup_datetime"] > f"{year}-%02d-01 00:00:00" % month]
    manhattan_taxi_data = manhattan_taxi_data[manhattan_taxi_data["tpep_dropoff_datetime"] > f"{year}-%02d-01 00:00:00" % month]
    manhattan_taxi_data = manhattan_taxi_data[manhattan_taxi_data["tpep_pickup_datetime"] < f"{year}-%02d-31 23:59:59" % month]
    manhattan_taxi_data = manhattan_taxi_data[manhattan_taxi_data["tpep_dropoff_datetime"] < f"{year}-%02d-31 23:59:59" % month]
    manhattan_taxi_data['tpep_pickup_datetime'] = pd.to_datetime(manhattan_taxi_data['tpep_pickup_datetime'])
    manhattan_taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(manhattan_taxi_data['tpep_dropoff_datetime'])
    manhattan_taxi_data['alignedtime'] = manhattan_taxi_data.tpep_pickup_datetime.dt.floor(f'{TIME_GAP*60}min')
    manhattan_taxi_data = manhattan_taxi_data.sort_values('alignedtime')
    
    time = pd.DataFrame()
    if month == 12:
        time = pd.DataFrame({'time':pd.date_range(f'{year}-12', f'{year+1}-1', freq=f'{TIME_GAP*60}min')})
    else:
        time = pd.DataFrame({'time':pd.date_range(f'{year}-%02d' % month, f'{year}-%02d' % (month+1), freq=f'{TIME_GAP*60}min')})
    time = time[: -1]
    group_time_inflow = manhattan_taxi_data.groupby(['alignedtime', 'DOLocationID']).size().reset_index(name='counts')
    group_time_outflow = manhattan_taxi_data.groupby(['alignedtime', 'PULocationID']).size().reset_index(name='counts')
    group_time_odflow = manhattan_taxi_data.groupby(['alignedtime', 'PULocationID', 'DOLocationID']).size().reset_index(name='counts')

    for index, row in time.iterrows():
        timeslot = row['time']
        inflow_data = group_time_inflow[group_time_inflow["alignedtime"] == timeslot]
        for i, row_ in inflow_data.iterrows():
            zone_id = manhattan_zones[manhattan_zones['zone_id'] == row_['DOLocationID']].graph_id
            count = row_['counts']
            in_outflow[0][index][zone_id] = count
        outflow_data = group_time_outflow[group_time_outflow["alignedtime"] == timeslot]
        for i, row_ in outflow_data.iterrows():
            zone_id = manhattan_zones[manhattan_zones['zone_id'] == row_['PULocationID']].graph_id
            count = row_['counts']
            in_outflow[1][index][zone_id] = count
        odflow_data = group_time_odflow[group_time_odflow["alignedtime"] == timeslot]
        for i, row_ in odflow_data.iterrows():
            o_id = manhattan_zones[manhattan_zones['zone_id'] == row_['PULocationID']].graph_id
            d_id = manhattan_zones[manhattan_zones['zone_id'] == row_['DOLocationID']].graph_id
            count = row_['counts']
            odflow[index, o_id, d_id] = count
            
    f = f'/home/cseadmin/mhy/data/NYCTaxiData/npyData/halfhour/yellow_tripdata_{year}-%02d-graph-inoutflow.npz' % month
    np.savez_compressed(f, in_outflow)

    f = f'/home/cseadmin/mhy/data/NYCTaxiData/npyData/halfhour/yellow_tripdata_{year}-%02d-graph-odflow.npz' % month
    np.savez_compressed(f, odflow)
    print(f"{year}-{month}Saved")

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=24)
    for year in [2019, 2020]:
        for month in range(1, 13):
            print(year, month)
            pool.apply_async(get_flow, (year, month, ))
    pool.close()
    pool.join()
    print("All subprocess done")

    
            
