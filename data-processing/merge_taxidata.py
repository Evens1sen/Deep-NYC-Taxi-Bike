import pandas as pd
import numpy as np
from datetime import datetime

TIME_START = 20190101000000  # %Y%m
TIME_END = 20201231235959  # %Y%m
TIME_STEP = 60  # minutes

filepath_list = []
for i in range(1, 13):
    filepath_list.append('/home/cseadmin/mhy/data-NYCTaxi/{}min/yellow_tripdata_2019-{}-graph-inoutflow.npz'.format(TIME_STEP, str(i).zfill(2)))
for i in range(1, 13):
    filepath_list.append('/home/cseadmin/mhy/data-NYCTaxi/{}min/yellow_tripdata_2020-{}-graph-inoutflow.npz'.format(TIME_STEP, str(i).zfill(2)))

out_flow = pd.DataFrame({'time':pd.date_range(str(TIME_START), str(TIME_END), freq='60min')})
in_flow = pd.DataFrame({'time':pd.date_range(str(TIME_START), str(TIME_END), freq='60min')})
for i in range(69):
    in_flow[str(i)] = 0
    out_flow[str(i)] = 0

days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
       31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,]
cur = 0
for i in range(len(filepath_list)):
    data = np.load(filepath_list[i])['arr_0']
    in_flow.iloc[[i for i in range(cur, cur+days[i]*24 )],[i for i in range(1,70)]] = pd.DataFrame(data[0,0:days[i]*24,0:69])
    out_flow.iloc[[i for i in range(cur, cur+days[i]*24 )],[i for i in range(1,70)]] = pd.DataFrame(data[1,0:days[i]*24,0:69])
    cur = cur +days[i]*24

in_flow = np.array(in_flow[in_flow.columns[1:]])
out_flow = np.array(out_flow[out_flow.columns[1:]])
in_outflow = np.zeros((in_flow.shape[0], in_flow.shape[1],  2))
in_outflow[:, :, 0] = in_flow.copy()
in_outflow[:, :, 1] = out_flow.copy()

with open('/home/cseadmin/mhy/data-NYCTaxi/test/{}-{}-NYCtaxi-inflow.npz'.format( str(TIME_START)[0:6], str(TIME_END)[0:6]),
          'wb') as f:
    np.savez_compressed(f, in_flow)
with open('/home/cseadmin/mhy/data-NYCTaxi/test/{}-{}-NYCtaxi-outflow.npz'.format( str(TIME_START)[0:6], str(TIME_END)[0:6]),
          'wb') as f:
    np.savez_compressed(f, out_flow)
with open(
        '/home/cseadmin/mhy/data-NYCTaxi/test/{}-{}-NYCtaxi-inoutflow.npz'.format(str(TIME_START)[0:6], str(TIME_END)[0:6]),
        'wb') as f:
    np.savez_compressed(f, in_outflow)

# with open('./data-NYCTaxi/{}min/{}-{}-NYCtaxi-inflow.npz'.format(TIME_STEP,str(TIME_START)[0:6],str(TIME_END)[0:6]), 'wb') as f:
#         np.savez_compressed(f, in_flow)
# with open('./data-NYCTaxi/{}min/{}-{}-NYCtaxi-outflow.npz'.format(TIME_STEP,str(TIME_START)[0:6],str(TIME_END)[0:6]), 'wb') as f:
#         np.savez_compressed(f, out_flow)
# with open('./data-NYCTaxi/{}min/{}-{}-NYCtaxi-inoutflow.npz'.format(TIME_STEP,str(TIME_START)[0:6],str(TIME_END)[0:6]), 'wb') as f:
#         np.savez_compressed(f, in_outflow)