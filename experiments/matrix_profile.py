#!pip install matrixprofile
import matrixprofile as mp
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
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
        if breaks is not None:
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
        data_len = 3000
        data, classes = f(round=1)
        window_size = 50
        profile = mp.compute(data, window_size)
        profile = mp.discover.discords(profile)
        mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
        axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
        axes[0].set_title('Raw Data', size=22)
        axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
        axes[1].set_title('Matrix Profile', size=22)
        for discord in profile['discords']:
            x = discord
            y = profile['mp'][discord]
            axes[1].plot(x, y, marker='*', markersize=10, c='r')
        plt.show()

    data_len = 5000
    split_len = 50
    data, classes = gen_four_segments_simple(data_len=data_len, round=1)
    window_size = 50
    profile = mp.compute(data, window_size)
    profile = mp.discover.discords(profile)
    mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
    axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
    axes[0].set_title('Raw Data', size=22)
    axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
    axes[1].set_title('Matrix Profile', size=22)
    for discord in profile['discords']:
        x = discord
        y = profile['mp'][discord]
        axes[1].plot(x, y, marker='*', markersize=10, c='r')

    plt.show()

    data_len = 5000
    split_len = 50
    data, classes = gen_four_segments_variance_only(data_len=data_len, round=1)
    window_size = 50
    profile = mp.compute(data, window_size)
    profile = mp.discover.discords(profile)
    mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
    axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
    axes[0].set_title('Raw Data', size=22)
    axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
    axes[1].set_title('Matrix Profile', size=22)
    for discord in profile['discords']:
        x = discord
        y = profile['mp'][discord]
        axes[1].plot(x, y, marker='*', markersize=10, c='r')

    plt.show()

    data_len = 5000
    split_len = 50
    data, classes = gen_four_segments_smooth_change(data_len=data_len, round=1)
    window_size = 50
    profile = mp.compute(data, window_size)
    profile = mp.discover.discords(profile)
    mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
    axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
    axes[0].set_title('Raw Data', size=22)
    axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
    axes[1].set_title('Matrix Profile', size=22)
    for discord in profile['discords']:
        x = discord
        y = profile['mp'][discord]
        axes[1].plot(x, y, marker='*', markersize=10, c='r')

    plt.show()

    data_len = 5000
    split_len = 50
    data, classes = gen_arma_data(data_len=data_len, round=1)
    window_size = 50
    profile = mp.compute(data, window_size)
    profile = mp.discover.discords(profile)
    mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
    axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
    axes[0].set_title('Raw Data', size=22)
    axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
    axes[1].set_title('Matrix Profile', size=22)
    for discord in profile['discords']:
        x = discord
        y = profile['mp'][discord]
        axes[1].plot(x, y, marker='*', markersize=10, c='r')

    plt.show()

    sys.path.append('../data/covid')
    from covid_data import get_covid_data
    split_len = 50
    data, classes = get_covid_data()
    window_size = 10
    profile = mp.compute(data, window_size)
    profile = mp.discover.discords(profile)
    mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
    axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
    axes[0].set_title('Raw Data', size=22)
    axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
    axes[1].set_title('Matrix Profile', size=22)
    for discord in profile['discords']:
        x = discord
        y = profile['mp'][discord]
        axes[1].plot(x, y, marker='*', markersize=10, c='r')

    plt.show()