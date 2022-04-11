MY_RES = './result/my/result.json'
SKL_RES = './result/sklearn/result.json'

import json
import matplotlib.pyplot as plt
import numpy as np

my_logs = json.load(open(MY_RES, 'r'))
skl_logs = json.load(open(SKL_RES, 'r'))

for k, vmy in my_logs.items():
    vskl = skl_logs[k]
    my_times = []
    skl_times = []
    ds = []
    for i, mylog in enumerate(vmy):
        skllog = vskl[i]
        my_times.append(mylog['avg_run_time'])
        skl_times.append(skllog['avg_run_time'])
        ds.append(i+1)
    ds = np.array(ds)
    plt.figure(figsize=(5, 4))
    plt.cla()
    plt.clf()
    plt.title("\"" + k + "\" clustering method run time statistics\n(average of 20 runs)")
    plt.bar(ds-0.1, my_times, width=0.2, label='My')
    plt.bar(ds+0.1, skl_times, width=0.2, label='Scikit-Learn')
    plt.xticks(ds)
    plt.xlabel('Data set number')
    plt.ylabel('Second(s)')
    plt.legend()
    plt.savefig('./result/' + k + '_perf.png', dpi=200)