import numpy as np
import matplotlib.pyplot as plt

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
        sc = plt.scatter(range(len(lst)), lst, c=classes, zorder=10, s=5, label=classes)
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