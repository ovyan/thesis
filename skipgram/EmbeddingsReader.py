import os

import numpy as np
import pandas as pd
import tensorflow as tf

from skipgram import DataProcessor

DIR = 'skipgram/graphs/'

class EmbeddingsReader():

    def extract_embeddings(self):
        # global names, features
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(DIR))
        sess = tf.Session()
        saver = tf.train.import_meta_graph(DIR + 'model.ckpt-1.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()
        embedding = graph.get_tensor_by_name("embedding:0")
        print(sess.run(embedding))
        vectors_np = sess.run(embedding)
        df_device_names = pd.DataFrame.from_csv(DIR + "metadata.tsv", sep="/t", header=None, index_col=None)
        device_names = df_device_names.values.ravel()
        print('device names values')
        print(device_names)
        # devices_dict = dict((v, i) for i, v in enumerate(device_names))
        # print('device dict - vocabulary built')
        # print(devices_dict)
        embed = tf.nn.embedding_lookup(embedding, device_names)
        print('Embedding lookup result')
        print(sess.run(embed))
        sorted_embeds = sess.run(embed)
        sorted_embeds_transposed = np.transpose(sorted_embeds)
        print(sorted_embeds_transposed.shape)
        print(sorted_embeds.shape)
        df = pd.DataFrame(data=sorted_embeds_transposed, columns=device_names)
        df.to_csv("energy_embeddings_gmm1.csv", index=False, encoding='utf-8')
        return df

reader = EmbeddingsReader()
reader.extract_embeddings()
