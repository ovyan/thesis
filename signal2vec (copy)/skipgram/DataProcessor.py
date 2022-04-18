import random
import pandas as pd
import numpy as np
import os
from sklearn.utils.extmath import cartesian

ZERO_STATE = 'zerostate'
NOISE = 'noise'
LOG_DIR = './graphs'


def extract_device_states(names):
    names_array = np.array(names, dtype=object)
    startstop_array = np.array(['start_', 'stop_'], dtype=object)
    devices = cartesian([startstop_array, names_array])
    devices = map(''.join, zip(devices[:, 0], devices[:, 1]))
    devices = np.append(devices, [ZERO_STATE, NOISE])
    devices = np.append(names_array, devices)
    return devices


def build_vocab(devices):
    devices_dict = dict((v, i) for i, v in enumerate(devices))
    metadata = os.path.join(LOG_DIR, 'metadata.tsv')
    with open(metadata, 'w') as metadata_file:
        for row in devices:
            metadata_file.write('%s\n' % row)
    return devices_dict


def build_devices_sequence(devices, devices_dict, names, pd_data):
    devices_names_sequence = list()
    devices_ids_sequence = list()
    for index, row in pd_data.iterrows():
        no_device = 1
        for i in range(2, len(names)):
            if row[i] != 0:
                no_device = 0
                pos = i - 2 + (row[i] - 1) * len(names)
                devices_ids_sequence.append(pos)
                devices_names_sequence.append(devices[pos])

        if row[1] == 0:
            devices_ids_sequence.append(devices_dict[ZERO_STATE])
            devices_names_sequence.append(ZERO_STATE)
        elif no_device:
            devices_ids_sequence.append(devices_dict[NOISE])
            devices_names_sequence.append(NOISE)
    print("Length of device names sequence")
    print(len(devices_names_sequence))
    return devices_ids_sequence, devices_names_sequence


def generate_sample(devices_ids_sequence, skip_window):
    print("length of devices ids")
    print(len(devices_ids_sequence))
    for index, center in enumerate(devices_ids_sequence):
        context = random.randint(1, skip_window)
        for target in devices_ids_sequence[max(0, index - context): index]:
            yield center, target
        for target in devices_ids_sequence[index + 1: index + context + 1]:
            yield center, target


def get_batch(iterator, batch_size):
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch


def get_device_states_size(file_path):
    pd_data = pd.read_csv(file_path, header=0)
    names = pd_data.columns.values[2:]
    devices = extract_device_states(names)
    return len(devices)


def process_data(file_path, batch_size, skip_window):
    devices_ids_sequence, _ = create_sequences(file_path)
    return generate_batches(batch_size, devices_ids_sequence, skip_window)


def process_gmm_data(file_path, batch_size, skip_window):
    df_data = pd.read_csv(file_path)
    energy_states_sequence = df_data.values.astype(int).flatten().tolist()
    return generate_batches(batch_size, energy_states_sequence, skip_window)

def get_number_of_energy_states(file_path):
    df_data = pd.read_csv(file_path)
    energy_states_sequence = df_data.values.astype(int).flatten().tolist()
    energy_set = set(energy_states_sequence)
    size = len(energy_set)
    print("Number of unique energy states: {}".format(size))
    return size


def generate_batches(batch_size, devices_ids_sequence, skip_window):
    pair_generator = generate_sample(devices_ids_sequence, skip_window)
    return get_batch(pair_generator, batch_size)


def create_sequences(file_path):
    pd_data = pd.read_csv(file_path, header=0)
    names = pd_data.columns.values[2:]
    devices = extract_device_states(names)
    devices_dict = build_vocab(devices)
    devices_ids_sequence, devices_names_sequence = build_devices_sequence(devices, devices_dict, names, pd_data)
    return devices_ids_sequence, devices_names_sequence
