import numpy as np
import csv

filepath_list = []
# for i in range(1, 13):
#     filepath_list.append('../data-NYCTaxi/60min/yellow_tripdata_2019-{}-graph-odflow.npz'.format(str(i).zfill(2)))
# for i in range(1, 13):
#     filepath_list.append('../data-NYCTaxi/60min/yellow_tripdata_2020-{}-graph-odflow.npz'.format(str(i).zfill(2)))
for i in range(1, 13):
    filepath_list.append('../data-NYCBike/60min/2019{}-citibike-tripdata-graph-odflow.npz'.format(str(i).zfill(2)))
for i in range(1, 13):
    filepath_list.append('../data-NYCBike/60min/2020{}-citibike-tripdata-graph-odflow.npz'.format(str(i).zfill(2)))
od_adjacency_matrix = np.zeros((69, 69), dtype='int32')
for i in range(len(filepath_list)):
    data = np.load(filepath_list[i])['arr_0']
    for j in range(744):
        od_adjacency_matrix = od_adjacency_matrix + data[j]

# od_adjacency_matrix = od_adjacency_matrix + od_adjacency_matrix.T
# od_adjacency_matrix = np.log(od_adjacency_matrix)
mask = (od_adjacency_matrix != 0)
od_adjacency_matrix[od_adjacency_matrix == 0] = np.inf
# od_adjacency_matrix[mask] = np.log(od_adjacency_matrix[mask])
od_adjacency_matrix[mask] = np.max(od_adjacency_matrix[mask])/od_adjacency_matrix[mask]

# od_adjacency_matrix [mask]= od_adjacency_matrix[mask]/np.std(od_adjacency_matrix[mask], ddof=1)

# f = open('../data-NYCZones/adjmatrix/W_od_taxi_new2.csv', 'w', encoding='utf-8')
f = open('../data-NYCZones/adjmatrix/W_od_bike_new2.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
id = []
for i in range(69):
    id.append(i)
csv_writer.writerow(id)
for i in od_adjacency_matrix:
    csv_writer.writerow(i)
f.close()



#%%


