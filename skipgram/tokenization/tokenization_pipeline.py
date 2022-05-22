import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

from tokenization.BatchGenerator import BatchGenerator

ENERGY_TOKENS_SEQUENCE = "energy_tokens_sequence.csv"
SAVED_MODEL = "gmm.pkl"

INPUT = '../../../Datasets/UKDALE/ukdale.h5'
WINDOW = 1


# start_date = "31-12-2015"
# end_date = "31-12-2016"


def generate_batches(start_date, end_date, window, building, dataset):
    tokenizer = BatchGenerator(start_date, end_date, window, building, dataset)
    # tokenizer.execute()
    return tokenizer.get_filename()


sequence_filename1_1 = generate_batches(start_date="1-3-2013", end_date="31-12-2013",
                                        window=WINDOW, building=1, dataset=INPUT)
sequence_filename1_2 = generate_batches(start_date="1-3-2014", end_date="31-12-2014",
                                        window=WINDOW, building=1, dataset=INPUT)
sequence_filename1_3 = generate_batches(start_date="1-3-2015", end_date="31-12-2015",
                                        window=WINDOW, building=1, dataset=INPUT)
sequence_filename1_4 = generate_batches(start_date="1-3-2016", end_date="31-12-2016",
                                        window=WINDOW, building=1, dataset=INPUT)
sequence_filename2 = generate_batches(None, None,
                                      window=WINDOW, building=2, dataset=INPUT)
sequence_filename3 = generate_batches(None, None,
                                      window=WINDOW, building=3, dataset=INPUT)

# df1_1 = pd.read_csv(sequence_filename1_1)
# df1_2 = pd.read_csv(sequence_filename1_2)
# df1_3 = pd.read_csv(sequence_filename1_3)
# df1_4 = pd.read_csv(sequence_filename1_4)
# df2 = pd.read_csv(sequence_filename2)
# df3 = pd.read_csv(sequence_filename3)
#
#
# def drop_values(df):
#     d = int(len(df) * 0.3)
#     print("number of d: {}".format(d))
#     df.drop(df.index[:d], inplace=True)
#     df.drop(df.index[-d:], inplace=True)
#

# drop_values(df1_1)
# drop_values(df1_2)
# drop_values(df1_3)
# drop_values(df1_4)
# drop_values(df2)
# drop_values(df3)

# df = df1_1
# df = df.append(df1_2, ignore_index=True)
# df = df.append(df1_3, ignore_index=True)
# df = df.append(df1_4, ignore_index=True)
# df = df.append(df2, ignore_index=True)
# df = df.append(df3, ignore_index=True)
# # print(df)
# print(df.shape)
# data = df.iloc[:, 1:]
# # print(data)
#
# model = GaussianMixture(n_components=1000, verbose=2, max_iter=1000, random_state=27)
# model.fit(data)
# joblib.dump(model, SAVED_MODEL)
#
# model1 = joblib.load(SAVED_MODEL)
# predictions = model1.predict(data)
# np.savetxt(ENERGY_TOKENS_SEQUENCE, predictions.astype(int))
