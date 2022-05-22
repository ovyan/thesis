from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
import faiss


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=100):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


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
        plt.figure(figsize=(12, 7))
        plt.plot(lst, alpha=0.7)
        sc = plt.scatter(range(len(lst)), lst, c=classes, zorder=10, s=25, label=classes)
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
        
        plt.title("Synthetic Simple")
    else:
        plt.plot(lst)
    plt.xlabel("t")
    plt.ylabel("$Y_t$")
    plt.show()


def split_list_by_n_elements(lst, n=100):
    return [lst[i:min(i + n, len(lst))] for i in range(0, len(lst), n)]


if __name__ == "__main__":
    data_len = 10_000
    split_len = 50

    data, classes = gen_four_segments_smooth_change(data_len=data_len, round=1)
    viz_list(data, classes=classes)
    data = list(map(str, data))
    data_split = split_list_by_n_elements(data, n=split_len)

    vector_size = 10
    window = 50
    model = Word2Vec(sentences=data_split, vector_size=vector_size, window=window, min_count=3, workers=4)
    model.wv.save_word2vec_format('model_name.bin', binary=True)
    vocab = list(model.wv.key_to_index)
    X = model.wv[vocab]

    vectors = []
    classes_per_vectors = []
    for i in range(0, len(data), split_len):
    sentence = data[i:min(len(data), i+split_len)]
    curr_classes = classes[i:min(len(data), i+split_len)]
    curr_vector = np.array([.0] * vector_size)
    curr_vector_N = 0
    for w in sentence:
        try:
            curr_vector += model.wv[w]
            curr_vector_N += 1
        except:
            pass
    if curr_vector_N != 0:
        curr_vector /= curr_vector_N
        classes_per_vectors.append(int(np.median(curr_classes)))
        vectors.append(curr_vector)

    vectors = np.array(vectors).astype(np.float32)
    from sklearn.metrics import silhouette_score
    import seaborn as sns

    for cluster_num in range(2, 10):
    kmeans_faiss = FaissKMeans(n_clusters=cluster_num)
    kmeans_faiss.fit(vectors)

    kmeans_faiss = FaissKMeans(n_clusters=6)
    kmeans_faiss.fit(vectors)
    pred_label = kmeans_faiss.predict(vectors).ravel()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(vectors)
    df = pd.DataFrame(X_pca, index=classes_per_vectors, columns=['x', 'y'])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(df['x'], df['y'], label=list(df.index), c=df.index)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)

    plt.title("PCA Ground truth")
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.show()


    X_pca = pca.fit_transform(vectors)
    df = pd.DataFrame(X_pca, index=classes_per_vectors, columns=['x', 'y'])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(df['x'], df['y'], label=list(df.index), c=pred_label)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)

    X_pca = pca.transform(kmeans_faiss.cluster_centers_)
    df = pd.DataFrame(X_pca, index=list(range(0, len(X_pca))), columns=['x', 'y'])
    scatter = ax.scatter(df['x'], df['y'], c=[3, 1, 0, 2, 4, 5], label=list(df.index), s=300, edgecolors='red')
    ax.add_artist(legend1)
    plt.title("PCA Predicted")
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.show()