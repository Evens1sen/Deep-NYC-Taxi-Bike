import csv

MODEL = 'GraphwaveNet'
CHANNEL = 4

line1 = []
line2 = []
line3 = []

for target in range(4):
    flow = ''
    if target == 0:
        flow = 'bikein'
    elif target == 1:
        flow = 'bikeout'
    elif target ==2:
        flow = 'taxiin'
    else:
        flow = 'taxiout'
    path = f'./{MODEL}_60min_TIMESTAMP33_{flow}.log'
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

# if CHANNEL == 'od':
#     for target in range(4):
#         path = f'./{MODEL}_60min_od_{target}.log'
#         hour1 = []
#         hour2 = []
#         hour3 = []
#         with open(path, 'r') as f:
#             for line in f:
#                 if '1 step' in line:
#                     hour1 = line.split(',')
#                     hour1 = [float(i.strip()) for i in hour1[-3:]]
#                 if '2 step' in line:
#                     hour2 = line.split(',')
#                     hour2 = [float(i.strip()) for i in hour2[-3:]]
#                 if '3 step' in line:
#                     hour3 = line.split(',')
#                     hour3 = [float(i.strip()) for i in hour3[-3:]]
#
#         line1.extend([hour1[-1], hour2[-1], hour3[-1]])
#         line2.extend([hour1[-2], hour2[-2], hour3[-2]])
#         line3.extend([hour1[-3], hour2[-3], hour3[-3]])
# else:
#     for target in range(CHANNEL):
#         flow = ''
#         if target == 0:
#             flow = 'bikein'
#         elif target == 1:
#             flow = 'bikeout'
#         elif target ==2:
#             flow = 'taxiin'
#         else:
#             flow = 'taxiout'
#         path = f'./{MODEL}_60min_1to1_{flow}.log'
#         hour1 = []
#         hour2 = []
#         hour3 = []
#         with open(path, 'r') as f:
#             for line in f:
#                 if '1 step' in line:
#                     hour1 = line.split(',')
#                     hour1 = [float(i.strip()) for i in hour1[-3:]]
#                 if '2 step' in line:
#                     hour2 = line.split(',')
#                     hour2 = [float(i.strip()) for i in hour2[-3:]]
#                 if '3 step' in line:
#                     hour3 = line.split(',')
#                     hour3 = [float(i.strip()) for i in hour3[-3:]]
#
#
#         line1.extend([hour1[-1], hour2[-1], hour3[-1]])
#         line2.extend([hour1[-2], hour2[-2], hour3[-2]])
#         line3.extend([hour1[-3], hour2[-3], hour3[-3]])

with open('result1.csv', mode='w') as f:
    csv_writer = csv.writer(f, delimiter=',')

    csv_writer.writerow(line1)
    csv_writer.writerow(line2)
    csv_writer.writerow(line3)