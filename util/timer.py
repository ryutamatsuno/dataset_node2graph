import time
from typing import Callable

__timekeys = {}


def start(key='') -> None:
    __timekeys[key] = time.time()


def pause(key='') -> None:
    start('__pause::' + key)


def resume(key='') -> None:
    x = stop('__pause::' + key)
    __timekeys[key] += x


def stop(key='') -> float:
    # if not key in __timekeys:
    #     raise ValueError(key + ' is not started')
    t = time.time() - __timekeys[key]
    del __timekeys[key]
    return t


def time2str(sec: float):
    # sec = int(sec+0.5)
    txt = ''
    if sec > 0:
        s = sec % 60
        sec = sec // 60
        txt = '%5.2f[s]' % (s)
    if sec > 0:
        m = sec % 60
        sec = sec // 60
        txt = '%2d[m]' % (m) + txt
    if sec > 0:
        h = sec % 24
        sec = sec // 24
        txt = '%2d[h]' % (h) + txt
    # print(sec)
    if sec > 0:
        d = sec
        # d = sec % 365
        # sec = sec //365
        txt = '%2d[d]' % (d) + txt
    # if sec > 0:
    #     y = sec
    #     txt = '%2d[y]' % (y) + txt

    return txt


def running_time(f: Callable, n_times: int = 1) -> float:
    ts = []

    for _ in range(n_times):
        start = time.time()
        f()
        ts.append(time.time() - start)

    return sum(ts) / n_times
