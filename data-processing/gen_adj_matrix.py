import csv
import numpy as np

f = open('./data-processing/W_NYC.csv', 'w', encoding='utf-8')

csv_writer = csv.writer(f)
id = []
for i in range(69):
    id.append(i)
csv_writer.writerow(id)

adj_matrix = np.load('./data-processing/adjacency_matrix_new.npy')
for i in adj_matrix:
    csv_writer.writerow(i)
f.close()
print('done')