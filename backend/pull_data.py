import json
import requests
from tqdm import tqdm
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_agged_trace(start_ts, end_ts):
    print(start_ts)
    traces = requests.get('http://106.75.2.46:16686/api/traces', params={
        'start': start_ts * 1000000,
        'end': end_ts * 1000000,
        'limit': 40,
        'lookback': 'custom',
        'maxDuration': '',
        'minDuration': '',
        'service': 'TiDB',
        'operation': 'session.runStmt',
    }).json()['data']
    new_traces = []

    # http://106.75.2.46:16686/api/traces?end=1543676199335000&limit=20&lookback=1h&maxDuration&minDuration&service=TiDB&start=1543672599335000
    for trace in traces:
        trace_ = {}
        trace_['sql'] = [s for s in trace['spans'] if s['operationName']
                         == 'session.runStmt'][0]['logs'][0]['fields'][0]['value']
        sql = trace_['sql']

        t = 0
        if re.search('select id from my_user where id=(.*)', sql):
            t = 1
        elif re.search('select count\(\*\) from my_user where age=(.*)', sql):
            t = 2
        elif re.search('insert (.*)', sql):
            t = 3
        trace_['sql_type'] = t
        for s in trace['spans']:
            trace_[s['operationName']] = s['duration']
        new_traces.append(trace_)

    def agg_traces(traces, sql_type):
        traces = [t for t in traces if t['sql_type'] == sql_type]
        if len(traces) > 0:
            traces_ = {
                'sql_type': sql_type,
                'sample': random.choice(traces),
            }
            for k in traces[0]:
                if type(traces[0][k]) == int:
                    l = [t.get(k, 0) for t in traces]
                    traces_[k] = sum(l) / len(l)
            return traces_
        return None

    return agg_traces(new_traces, 1), agg_traces(new_traces, 2), agg_traces(new_traces, 3)


def get_cpu_metrics(start_ts, end_ts):
    data = requests.get(
        'http://106.75.2.46:9090/api/v1/query_range?query=100%20-%20(avg%20by%20(instance)%20(irate(node_cpu_seconds_total{job=%22node_exporter%22,mode=%22idle%22}[30s]))%20*%20100)%20&start=' + str(
            start_ts) + '&end=' + str(end_ts) + '&step=1').json()['data']
    return data['result'][0]['values'][0][1]


start_ts = 1543657740
INTERVAL = 10
STEPS = 10
data = [None for x in range(STEPS)]

with ThreadPoolExecutor(max_workers=50) as executor:
    future_to_url = {executor.submit(
        load_agged_trace, start_ts + i * INTERVAL, start_ts + i * INTERVAL + INTERVAL): i for i in range(STEPS)}
    for future in tqdm(as_completed(future_to_url)):
        i = future_to_url[future]
        data[i] = future.result()
# json.dump(data, open('result_{}.json'.format(start_ts), 'w'))

# with ThreadPoolExecutor(max_workers=100) as executor:
#     future_to_url = {executor.submit(
#         get_cpu_metrics, start_ts + i * INTERVAL, start_ts + i * INTERVAL + INTERVAL): i for i in range(STEPS)}
#     for future in tqdm(as_completed(future_to_url)):
#         i = future_to_url[future]
#         try:
#             data[i] = future.result()
#         except Exception as exc:
#             pass
# json.dump(data, open('result_cpus_{}.json'.format(start_ts), 'w'))
