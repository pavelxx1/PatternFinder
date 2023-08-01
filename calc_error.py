# import pandas as pd
import numpy as np
# from sklearn import preprocessing
# _scaler = preprocessing.MinMaxScaler(feature_range=(0,10))

# def norm(a):
#     return _scaler.fit_transform(np.array(a).reshape(-1,1)).flatten()

def pct_from_first(arr):
    arr[arr == 0] = 1
    return (arr - arr[0]) / arr[0]

def _pct(arr: np.ndarray, n: int = 1) -> np.ndarray:
    arr[arr == 0] = 1
    out = np.full_like(arr, 0, dtype=float)
    out[n:] = arr[n:] / arr[:-n] - 1
    return np.nan_to_num(out) 

def norm(data,do=True):
    if do:
        _min = np.min(data)
        _max = np.max(data)
        return (data - _min) / (_max - _min)
    else: return np.array(data)

def multi_calc_total_error(metric_func,pattrn,stack,sz,horizon,step,do,pct,queue):
    err_list = []
    check_len = len(stack[-1])-sz-horizon
    for j in range(0,check_len,step):
        error = 0.0
        # __list = []
        for x in range(len(stack)-1):
            x1 = pattrn[x]
            x2 = stack[x][j:j+sz]
            # __list.append({f'x{x}':pct_from_first(np.array(x1)),f'y{x}':pct_from_first(np.array(x2))})# ,'all':__list
            if not 0xBAD in x2: # array has our marker!
                if not pct: error += np.nan_to_num(metric_func(norm(x1,do),norm(x2,do)))
                else: error += np.nan_to_num(metric_func(pct_from_first(np.array(x1)),pct_from_first(np.array(x2))))
        if error:
            err_list.append({'index':int(stack[-1][j]),
                             'error':error})
    queue.put(err_list)
