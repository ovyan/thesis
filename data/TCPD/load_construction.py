from load_dataset import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import json

def get_construction_data(round=1):
    ts = TimeSeries.from_json('./datasets/construction/construction.json')
    data = ts.df['Total Private Construction Spending'].values
    annotations = None
    with open('annotations.json') as f:
        d = json.load(f)
        annotations = d['construction']
        annotations = annotations["6"]
    
    prev = 0
    marker = 0
    classes = []
    for elem in annotations:
        diff = elem - prev
        classes += [marker] * diff
        prev = elem
        marker += 1

    classes += [marker] * (len(data) - annotations[-1])
    return list(np.round(data, round)), classes

def viz_list(lst, classes=None):
    if classes is not None:
        plt.figure(figsize=(10, 7))
        plt.plot(lst)
        sc = plt.scatter(range(len(lst)), lst, c=classes, zorder=10, s=15, label=classes)
        plt.legend(handles=sc.legend_elements()[0], labels=list(set(classes)), title="classes")
        vert_lines = []
        for uniq_val in set(classes):
            vert_lines.append(classes.index(uniq_val))
        for xc in vert_lines[1:]:
            plt.axvline(x=xc, linestyle='--', color='k')
    else:
        plt.plot(lst)
    plt.show()


if __name__ == "__main__":
    data, classes = get_construction_data()
    viz_list(data, classes)