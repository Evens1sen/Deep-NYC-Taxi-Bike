import csv

MODEL = 'DCRNN'
CHANNEL = 2
OD = False
MULTIGRAPH = False
TIMESTAMP = True

line1 = []
line2 = []
line3 = []

for target in range(4):
    if OD:
        path = f'./model/log/{MODEL}_od_{target}.log'
    elif MULTIGRAPH:
        path = f'./model/log/{MODEL}_multi_{target}.log'
    elif TIMESTAMP:
        path = f'./model/log/{MODEL}_TIMESTAMP_{target}.log'
    else:
        path = f'./model/log/{MODEL}_{CHANNEL}to1_{target}.log'
    hour1 = []
    hour2 = []
    hour3 = []
    with open(path, 'r') as f:
        for line in f:
            if '1 step' in line:
                hour1 = line.split(',')
                hour1 = [float(i.strip()) for i in hour1[-3:]]
            if '2 step' in line:
                hour2 = line.split(',')
                hour2 = [float(i.strip()) for i in hour2[-3:]]
            if '3 step' in line:
                hour3 = line.split(',')
                hour3 = [float(i.strip()) for i in hour3[-3:]]


    line1.extend([hour1[-1], hour2[-1], hour3[-1]])
    line2.extend([hour1[-2], hour2[-2], hour3[-2]])
    line3.extend([hour1[-3], hour2[-3], hour3[-3]])


with open(f'./model/log/{MODEL}_{CHANNEL}_reuslts.csv', mode='w') as f:
    print(f'Save result to ./model/log/{MODEL}_{CHANNEL}_reuslts.csv' )
    csv_writer = csv.writer(f, delimiter=',')

    csv_writer.writerow(line1)
    csv_writer.writerow(line2)
    csv_writer.writerow(line3)