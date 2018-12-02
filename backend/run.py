import time
from threading import Thread

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm_notebook as tqdm
import random
import string
from multiprocessing.pool import ThreadPool
import pymysql.cursors

# Connect to the database
# conn = pymysql.connect(host='106.75.2.46',
#                        port=4000,
#                        user='root',
#                        password='',
#                        db='my_tidb', autocommit=True)

# c = cursor.execute('select * from my_user where id=%s',[10])

p_index_select = 1
p_agg_select = 1
p_insert = 0.4
p_rate = 0.3


def run(x):
    conn = pymysql.connect(host='106.75.2.46',
                           port=4000,
                           user='root',
                           password='',
                           db='my_tidb', autocommit=True)
    cursor = conn.cursor()
    while True:
        if random.random() < p_agg_select * p_rate:
            age = random.randint(18, 90)
            c = cursor.execute('select count(*) from my_user where age=%s', [age])
        if random.random() < p_index_select * p_rate:
            id = random.randint(18, 90)
            c = cursor.execute('select id from my_user where id=%s', [id])
        if random.random() < p_insert * p_rate:
            age = random.randint(18, 90)
            name = ''.join(random.choice(string.ascii_uppercase + string.digits)
                           for _ in range(5))
            addr = ''.join(random.choice(string.ascii_uppercase + string.digits)
                           for _ in range(10))
            c = cursor.execute(
                'insert into my_user (name, age, addr) values (%s, %s, %s)', [name, age, addr])
        time.sleep(2)


def f(x):
    x = x / 2 / 3.14

    base1 = np.sin(x / 5)
    noise = np.sin(x) * 0.3 + np.sin(5 * x) * 0.5 + 3 + np.random.rand() * 2

    if base1 > 0:
        base1 = base1 * 2
    else:
        base1 = base1 / 2
        noise *= 2
    base1 = base1 + 0.7
    return (noise * base1 + np.random.rand() * 2) / 16


def update_prate():
    global p_rate
    i = 0
    while True:
        i += 1
        p_rate = f(i)
        time.sleep(1)
        print('p_rate', p_rate)


Thread(target=update_prate).start()

CORE_NUMBERS = 120
NUMBERS = range(CORE_NUMBERS)
pool = ThreadPool(CORE_NUMBERS)
pool.map(run, NUMBERS)
