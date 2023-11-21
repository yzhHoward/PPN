import os
import numpy as np
import pickle
import math

from utils.utils import load_data
from utils.readers import InHospitalMortalityReader
from utils.readers import PhenotypingReader
from utils.preprocessing import Discretizer, Normalizer

demographic_data = []
diagnosis_data = []
idx_list = []
train_reader = None
val_reader = None
test_reader = None
small_part = False
discretizer = None
normalizer = None


def mimic_init(type='mortality'):
    global demographic_data, diagnosis_data, idx_list, train_reader, val_reader, test_reader, discretizer, normalizer

    if type == 'mortality':
        data_path = './data/MIMIC-III/mortality/'
        train_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(data_path, 'train'),
            listfile=os.path.join(data_path, 'train_listfile.csv'),
            period_length=48.0)
        val_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(data_path, 'train'),
            listfile=os.path.join(data_path, 'val_listfile.csv'),
            period_length=48.0)
        test_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(data_path, 'test'),
            listfile=os.path.join(data_path, 'test_listfile.csv'),
            period_length=48.0)
    else:
        data_path = './data/MIMIC-III/phenotyping/'
        train_reader = PhenotypingReader(
            dataset_dir=os.path.join(data_path, 'train'),
            listfile=os.path.join(data_path, 'train_listfile.csv'))
        val_reader = PhenotypingReader(
            dataset_dir=os.path.join(data_path, 'train'),
            listfile=os.path.join(data_path, 'val_listfile.csv'))
        test_reader = PhenotypingReader(
            dataset_dir=os.path.join(data_path, 'test'),
            listfile=os.path.join(data_path, 'test_listfile.csv'))

    discretizer = Discretizer(timestep=1.0,
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')
    discretizer_header = discretizer.transform(
        train_reader.read_example(0)["X"])[1].split(',')
    print(discretizer_header)
    exit()
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]

    normalizer = Normalizer(
        fields=cont_channels)  # choose here which columns to standardize
    if type == 'mortality':
        normalizer_state = 'ihm_normalizer'
    else:
        normalizer_state = 'ph.normalizer'
    normalizer_state = os.path.join(os.path.dirname(data_path),
                                    normalizer_state)
    normalizer.load_params(normalizer_state)

    demo_path = data_path + 'demographic/'
    for cur_name in os.listdir(demo_path):
        cur_id, cur_episode = cur_name.split('_', 1)
        cur_episode = cur_episode[:-4]
        cur_file = demo_path + cur_name

        with open(cur_file, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            if header[0] != "Icustay":
                continue
            cur_data = tsfile.readline().strip().split(',')

        if len(cur_data) == 1:
            cur_demo = np.zeros(12)
            cur_diag = np.zeros(128)
        else:
            if cur_data[3] == '':
                cur_data[3] = 60.0
            if cur_data[4] == '':
                cur_data[4] = 160
            if cur_data[5] == '':
                cur_data[5] = 60

            cur_demo = np.zeros(12)
            cur_demo[int(cur_data[1])] = 1
            cur_demo[5 + int(cur_data[2])] = 1
            cur_demo[9:] = cur_data[3:6]
            cur_diag = np.array(cur_data[8:], dtype=np.int)

        demographic_data.append(cur_demo)
        diagnosis_data.append(cur_diag)
        idx_list.append(cur_id + '_' + cur_episode)

    for each_idx in range(9, 12):
        cur_val = []
        for i in range(len(demographic_data)):
            cur_val.append(demographic_data[i][each_idx])
        cur_val = np.array(cur_val)
        _mean = np.mean(cur_val)
        _std = np.std(cur_val)
        _std = _std if _std > 1e-7 else 1e-7
        for i in range(len(demographic_data)):
            demographic_data[i][each_idx] = (demographic_data[i][each_idx] -
                                             _mean) / _std


def get_demo(raw_name):
    demo = []
    for i in range(len(raw_name)):
        cur_id, cur_ep, _ = raw_name[i].split('_', 2)
        cur_idx = cur_id + '_' + cur_ep
        cur_demo = demographic_data[idx_list.index(cur_idx)]
        demo.append(cur_demo)

    return demo


def load_mimic_mortality():
    train_raw = load_data(train_reader,
                          discretizer,
                          normalizer,
                          small_part,
                          return_names=True)
    val_raw = load_data(val_reader,
                        discretizer,
                        normalizer,
                        small_part,
                        return_names=True)
    test_raw = load_data(test_reader,
                         discretizer,
                         normalizer,
                         small_part,
                         return_names=True)

    train_dataset = train_raw['data'][0], train_raw['data'][1], get_demo(
        train_raw['names'])
    valid_dataset = val_raw['data'][0], val_raw['data'][1], get_demo(
        val_raw['names'])
    test_dataset = test_raw['data'][0], test_raw['data'][1], get_demo(
        test_raw['names'])

    return train_dataset, valid_dataset, test_dataset


def load_mimic_mortality_file():
    return pickle.load(open('./data/MIMIC-III/mortality/data.pkl', 'rb'))


def load_mimic_mortality_few_visit(ratio=0.5):
    import random
    import torch
    lens = 48
    train_data, dev_data, test_data = pickle.load(
        open('./data/MIMIC-III/mortality/data.pkl', 'rb'))
    train_x, train_y, train_static = train_data
    dev_x, dev_y, dev_static = dev_data
    test_x, test_y, test_static = test_data
    train_x = list(train_x)
    dev_x = list(dev_x)
    test_x = list(test_x)
    # train_x = [x[:math.ceil(ratio * lens)] for x in train_x]
    # dev_x = [x[:math.ceil(ratio * lens)] for x in dev_x]
    # test_x = [x[:math.ceil(ratio * lens)] for x in test_x]

    for i in range(len(train_x)):
        index = torch.randperm(len(train_x[i])).tolist()[:math.ceil(ratio * len(train_x[i]))]
        index.sort()
        train_x[i] = [train_x[i][j] for j in index]
    for i in range(len(dev_x)):
        index = torch.randperm(len(train_x[i])).tolist()[:math.ceil(ratio * len(train_x[i]))]
        index.sort()
        dev_x[i] = [dev_x[i][j] for j in index]
    for i in range(len(test_x)):
        index = torch.randperm(len(train_x[i])).tolist()[:math.ceil(ratio * len(train_x[i]))]
        index.sort()
        test_x[i] = [test_x[i][j] for j in index]
    return (train_x, train_y, train_static), (dev_x, dev_y, dev_static), (test_x, test_y, test_static)


def load_mimic_mortality_sparse(ratio=0.5):
    import random

    train_data, dev_data, test_data = pickle.load(
        open('./data/MIMIC-III/mortality/data.pkl', 'rb'))
    train_x, train_y, train_static = train_data
    dev_x, dev_y, dev_static = dev_data
    test_x, test_y, test_static = test_data
    for i in range(len(train_x)):
        for j in range(len(train_x[i])):
            for k in range(len(train_x[i][j])):
                if random.random() >= ratio:
                    train_x[i][j][k] = 0
    for i in range(len(dev_x)):
        for j in range(len(dev_x[i])):
            for k in range(len(dev_x[i][j])):
                if random.random() >= ratio:
                    dev_x[i][j][k] = 0
    for i in range(len(test_x)):
        for j in range(len(test_x[i])):
            for k in range(len(test_x[i][j])):
                if random.random() >= ratio:
                    test_x[i][j][k] = 0
    return (train_x, train_y, train_static), (dev_x, dev_y, dev_static), (test_x, test_y, test_static)
