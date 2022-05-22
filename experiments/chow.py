#!pip install chowtest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
import chow_test
import sys


def gen_four_segments_simple(data_len=1000, round=3):
    segment_size = data_len // 4
    np.random.seed(42)
    first_segment = np.round(np.random.normal(10, 1, segment_size), round)
    second_segment = np.round(np.random.normal(20, 1, segment_size), round)
    third_segment = np.round(np.random.normal(20, 2, segment_size), round)
    fourth_segment = np.round(np.random.normal(30, 3, segment_size), round)
    classes = [0] * (data_len // 4) + [1] * (data_len // 4) + [2] * (data_len // 4) + [3] * (data_len // 4)

    return list(first_segment) + list(second_segment) + list(third_segment) + list(fourth_segment), classes


def gen_four_segments_variance_only(data_len=1000, round=3):
    segment_size = data_len // 4
    np.random.seed(42)
    first_segment = np.round(np.random.normal(0, 0.3, segment_size), round)
    second_segment = np.round(np.random.normal(0, 1, segment_size), round)
    third_segment = np.round(np.random.normal(0, 2, segment_size), round)
    fourth_segment = np.round(np.random.normal(0, 3, segment_size), round)
    classes = [0] * (data_len // 4) + [1] * (data_len // 4) + [2] * (data_len // 4) + [3] * (data_len // 4)

    return list(first_segment) + list(second_segment) + list(third_segment) + list(fourth_segment), classes


def gen_four_segments_smooth_change(data_len=1000, round=3):
    segment_size = data_len // 4
    smooth_segment_size = data_len // 4
    np.random.seed(42)
    first_segment = np.round(np.random.normal(10, 1, segment_size), round)
    smooth_segment_first = np.round(np.random.normal(0, 1, segment_size) + np.linspace(first_segment[-1], 20, smooth_segment_size), round)
    second_segment = np.round(np.random.normal(20, 1, segment_size), round)
    third_segment = np.round(np.random.normal(20, 2, segment_size), round)
    smooth_segment_third = np.round(np.random.normal(0, 2, segment_size) + np.linspace(third_segment[-1], 30, smooth_segment_size), round)
    fourth_segment = np.round(np.random.normal(30, 3, segment_size), round)
    classes = [0] * (data_len // 4) + [4] * (data_len // 4)  + [1] * (data_len // 4) + [2] * (data_len // 4) + [5] * (data_len // 4) + [3] * (data_len // 4)

    return list(first_segment) + list(smooth_segment_first) + list(second_segment) + list(third_segment) + list(smooth_segment_third) + list(fourth_segment), classes


def gen_arma_data(data_len=1000, round=3):
    np.random.seed(42)
    ar_coefs = [1, -1]
    ma_coefs = [-2, -0.8]
    y = np.round(arma_generate_sample(ar_coefs, ma_coefs, nsample=data_len // 2), round)

    ar_coefs = [1.01, -1]
    ma_coefs = [-2, -1]
    y_ = np.round(arma_generate_sample(ar_coefs, ma_coefs, nsample=data_len // 2), round)
    shift = y_[0] - y[-1]
    y_ -= shift
    classes = [0] * len(y) + [1] * len(y_)

    return list(y) + list(y_), classes


def viz_list(lst, classes=None, breaks=None):
    if classes is not None:
        plt.figure(figsize=(10, 7))
        plt.plot(lst)
        sc = plt.scatter(range(len(lst)), lst, c=classes, zorder=10, s=10, label=classes)
        plt.legend(handles=sc.legend_elements()[0], labels=list(set(classes)), title="classes")
        vert_lines = []
        if breaks is not None and len(breaks) > 0:
            for xc in breaks:
                plt.axvline(x=xc, linestyle='--', color='r', label='Chow Test')
        else:
            for uniq_val in set(classes):
                vert_lines.append(classes.index(uniq_val))
            for xc in vert_lines[1:]:
                plt.axvline(x=xc, linestyle='--', color='k')
        
        plt.title("Chow Test")
    else:
        plt.plot(lst)
    plt.xlabel("t")
    plt.ylabel("$Y_t$")
    plt.show()


def split_list_by_n_elements(lst, n=100):
    return [lst[i:min(i + n, len(lst))] for i in range(0, len(lst), n)]


def chow_test_res(first, second):
    df = pd.DataFrame({'x': list(range(0, len(first) + len(second))), 'y': first + second})
    res = chow_test.chow_test(
        X_series=df.x,
        y_series=df.y,
        last_index=len(first) - 1,
        first_index=len(first),
        significance=.05)

    is_significant = res[1] < .05
    return is_significant


if __name__ == "__main__":
    sys.path.append('../data/TCPD')
    from load_apple import get_apple_data
    from load_bitcoin import get_bitcoin_data
    from load_scanline_42049 import get_scanline_42049_data
    from load_construction import get_construction_data

    for f in [get_apple_data, get_bitcoin_data, get_scanline_42049_data, get_construction_data]:
        split_len = 50
        data, classes = f(round=1)
        breaks = []
        data_new = split_list_by_n_elements(data, n=split_len)
        for i in range(1, len(data_new)):
            is_significant = chow_test_res(data_new[i - 1], data_new[i])
            if is_significant:
                breaks.append(i * split_len)
        viz_list(data, classes=classes, breaks=breaks)

    data_len = 5000
    split_len = 50
    data, classes = gen_four_segments_simple(data_len=data_len, round=1)
    breaks = []
    data_new = split_list_by_n_elements(data, n=split_len)
    for i in range(1, len(data_new)):
        is_significant = chow_test_res(data_new[i - 1], data_new[i])
        if is_significant:
            breaks.append(i * split_len)
    viz_list(data, classes=classes, breaks=breaks)

    data_len = 5000
    split_len = 50
    data, classes = gen_four_segments_variance_only(data_len=data_len, round=1)
    breaks = []
    data_new = split_list_by_n_elements(data, n=split_len)
    for i in range(1, len(data_new)):
        is_significant = chow_test_res(data_new[i - 1], data_new[i])
        if is_significant:
            breaks.append(i * split_len)
    viz_list(data, classes=classes, breaks=breaks)

    data_len = 5000
    split_len = 50
    data, classes = gen_four_segments_smooth_change(data_len=data_len, round=1)
    breaks = []
    data_new = split_list_by_n_elements(data, n=split_len)
    for i in range(1, len(data_new)):
        is_significant = chow_test_res(data_new[i - 1], data_new[i])
        if is_significant:
            breaks.append(i * split_len)
    viz_list(data, classes=classes, breaks=breaks)

    data_len = 5000
    split_len = 50
    data, classes = gen_arma_data(data_len=data_len, round=1)
    breaks = []
    data_new = split_list_by_n_elements(data, n=split_len)
    for i in range(1, len(data_new)):
        is_significant = chow_test_res(data_new[i - 1], data_new[i])
        if is_significant:
            breaks.append(i * split_len)
    viz_list(data, classes=classes, breaks=breaks)

    sys.path.append('../data/covid')
    from covid_data import get_covid_data
    split_len = 50
    data, classes = get_covid_data()
    breaks = []
    data_new = split_list_by_n_elements(data, n=split_len)
    for i in range(1, len(data_new)):
        is_significant = chow_test_res(data_new[i - 1], data_new[i])
        if is_significant:
            breaks.append(i * split_len)
    viz_list(data, classes=classes, breaks=breaks)