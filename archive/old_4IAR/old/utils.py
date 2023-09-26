import datetime
import numpy as np
import os
import time

def printl(*args, flush=True, **kwargs):
    time_str = f'[{datetime.datetime.today()}]'
    print(time_str, flush=flush, *args, **kwargs)

def os_reseed():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))