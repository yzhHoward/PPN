from copy import deepcopy
import math
import pickle
import random


def load_challenge_new():
    data_path = './data/Challenge/new/'

    all_x = pickle.load(open(data_path + 'x.dat', 'rb'))
    all_y = pickle.load(open(data_path + 'y.dat', 'rb'))
    static = pickle.load(open(data_path + 'demo.dat', 'rb'))
    all_x_len = [len(i) for i in all_x]

    train_num = int(len(all_x) * 0.8) + 1
    dev_num = int(len(all_x) * 0.1) + 1
    test_num = int(len(all_x) * 0.1)
    assert (train_num + dev_num + test_num == len(all_x))

    train_x = []
    train_y = []
    train_static = []
    train_x_len = []
    for idx in range(train_num):
        train_x.append(all_x[idx])
        train_y.append(int(all_y[idx][-1]))
        train_static.append(static[idx])
        train_x_len.append(all_x_len[idx])

    dev_x = []
    dev_y = []
    dev_static = []
    dev_x_len = []
    for idx in range(train_num, train_num + dev_num):
        dev_x.append(all_x[idx])
        dev_y.append(int(all_y[idx][-1]))
        dev_static.append(static[idx])
        dev_x_len.append(all_x_len[idx])

    test_x = []
    test_y = []
    test_static = []
    test_x_len = []
    for idx in range(train_num + dev_num, train_num + dev_num + test_num):
        test_x.append(all_x[idx])
        test_y.append(int(all_y[idx][-1]))
        test_static.append(static[idx])
        test_x_len.append(all_x_len[idx])

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)

    # train_x = pickle.load(open(data_path + 'train_x.dat', 'rb'))
    # train_y = pickle.load(open(data_path + 'train_y.dat', 'rb'))
    # train_static = pickle.load(open(data_path + 'train_static.dat', 'rb'))
    # train_x_len = pickle.load(open(data_path + 'train_x_len.dat', 'rb'))

    # dev_x = pickle.load(open(data_path + 'dev_x.dat', 'rb'))
    # dev_y = pickle.load(open(data_path + 'dev_y.dat', 'rb'))
    # dev_static = pickle.load(open(data_path + 'dev_static.dat', 'rb'))
    # dev_x_len = pickle.load(open(data_path + 'dev_x_len.dat', 'rb'))

    # test_x = pickle.load(open(data_path + 'test_x.dat', 'rb'))
    # test_y = pickle.load(open(data_path + 'test_y.dat', 'rb'))
    # test_static = pickle.load(open(data_path + 'test_static.dat', 'rb'))
    # test_x_len = pickle.load(open(data_path + 'test_x_len.dat', 'rb'))

    return (train_x, train_y, train_x_len,
            train_static), (dev_x, dev_y, dev_x_len,
                            dev_static), (test_x, test_y, test_x_len,
                                          test_static)


def load_challenge_new_few_visit(ratio):
    import torch
    data_path = './data/Challenge/new/'

    all_x = pickle.load(open(data_path + 'x.dat', 'rb'))
    all_y = pickle.load(open(data_path + 'y.dat', 'rb'))
    static = pickle.load(open(data_path + 'demo.dat', 'rb'))
    all_x_len = [len(i) for i in all_x]

    train_num = int(len(all_x) * 0.8) + 1
    dev_num = int(len(all_x) * 0.1) + 1
    test_num = int(len(all_x) * 0.1)
    assert (train_num + dev_num + test_num == len(all_x))

    train_x = []
    train_y = []
    train_static = []
    train_x_len = []
    for idx in range(train_num):
        train_x.append(all_x[idx])
        train_y.append(int(all_y[idx][-1]))
        train_static.append(static[idx])
        train_x_len.append(all_x_len[idx])

    dev_x = []
    dev_y = []
    dev_static = []
    dev_x_len = []
    for idx in range(train_num, train_num + dev_num):
        dev_x.append(all_x[idx])
        dev_y.append(int(all_y[idx][-1]))
        dev_static.append(static[idx])
        dev_x_len.append(all_x_len[idx])

    test_x = []
    test_y = []
    test_static = []
    test_x_len = []
    for idx in range(train_num + dev_num, train_num + dev_num + test_num):
        test_x.append(all_x[idx])
        test_y.append(int(all_y[idx][-1]))
        test_static.append(static[idx])
        test_x_len.append(all_x_len[idx])

    for i in range(len(train_x)):
        index = torch.randperm(len(train_x[i])).tolist()[:math.ceil(ratio * len(train_x[i]))]
        index.sort()
        train_x[i] = [train_x[i][j] for j in index]
    for i in range(len(dev_x)):
        index = torch.randperm(len(dev_x[i])).tolist()[:math.ceil(ratio * len(dev_x[i]))]
        index.sort()
        dev_x[i] = [dev_x[i][j] for j in index]
    for i in range(len(test_x)):
        index = torch.randperm(len(test_x[i])).tolist()[:math.ceil(ratio * len(test_x[i]))]
        index.sort()
        test_x[i] = [test_x[i][j] for j in index]
    # train_x = [x[:math.ceil(ratio * len(x))] for x in train_x]
    train_x_len = [len(i) for i in train_x]
    # dev_x = [x[:math.ceil(ratio * len(x))] for x in dev_x]
    dev_x_len = [len(i) for i in dev_x]
    # test_x = [x[:math.ceil(ratio * len(x))] for x in test_x]
    test_x_len = [len(i) for i in test_x]

    return (train_x, train_y, train_x_len,
            train_static), (dev_x, dev_y, dev_x_len,
                            dev_static), (test_x, test_y, test_x_len,
                                          test_static)


def load_challenge_new_sparse(ratio):
    data_path = './data/Challenge/new/'

    all_x = pickle.load(open(data_path + 'x.dat', 'rb'))
    all_y = pickle.load(open(data_path + 'y.dat', 'rb'))
    static = pickle.load(open(data_path + 'demo.dat', 'rb'))
    all_x_len = [len(i) for i in all_x]

    train_num = int(len(all_x) * 0.8) + 1
    dev_num = int(len(all_x) * 0.1) + 1
    test_num = int(len(all_x) * 0.1)
    assert (train_num + dev_num + test_num == len(all_x))

    train_x = []
    train_y = []
    train_static = []
    train_x_len = []
    for idx in range(train_num):
        train_x.append(all_x[idx])
        train_y.append(int(all_y[idx][-1]))
        train_static.append(static[idx])
        train_x_len.append(all_x_len[idx])

    dev_x = []
    dev_y = []
    dev_static = []
    dev_x_len = []
    for idx in range(train_num, train_num + dev_num):
        dev_x.append(all_x[idx])
        dev_y.append(int(all_y[idx][-1]))
        dev_static.append(static[idx])
        dev_x_len.append(all_x_len[idx])

    test_x = []
    test_y = []
    test_static = []
    test_x_len = []
    for idx in range(train_num + dev_num, train_num + dev_num + test_num):
        test_x.append(all_x[idx])
        test_y.append(int(all_y[idx][-1]))
        test_static.append(static[idx])
        test_x_len.append(all_x_len[idx])

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

    return (train_x, train_y, train_x_len,
            train_static), (dev_x, dev_y, dev_x_len,
                            dev_static), (test_x, test_y, test_x_len,
                                          test_static)


def load_challenge_zerofill():
    data_path = './data/Challenge/normalized/'

    train_x = pickle.load(open(data_path + 'train_x.dat', 'rb'))
    train_y = pickle.load(open(data_path + 'train_y.dat', 'rb'))
    train_names = pickle.load(open(data_path + 'train_names.dat', 'rb'))
    train_static = pickle.load(open(data_path + 'train_static.dat', 'rb'))
    train_x_len = pickle.load(open(data_path + 'train_x_len.dat', 'rb'))
    train_mask_x = pickle.load(open(data_path + 'train_mask_x.dat', 'rb'))

    dev_x = pickle.load(open(data_path + 'dev_x.dat', 'rb'))
    dev_y = pickle.load(open(data_path + 'dev_y.dat', 'rb'))
    dev_names = pickle.load(open(data_path + 'dev_names.dat', 'rb'))
    dev_static = pickle.load(open(data_path + 'dev_static.dat', 'rb'))
    dev_x_len = pickle.load(open(data_path + 'dev_x_len.dat', 'rb'))
    dev_mask_x = pickle.load(open(data_path + 'dev_mask_x.dat', 'rb'))

    test_x = pickle.load(open(data_path + 'test_x.dat', 'rb'))
    test_y = pickle.load(open(data_path + 'test_y.dat', 'rb'))
    test_names = pickle.load(open(data_path + 'test_names.dat', 'rb'))
    test_static = pickle.load(open(data_path + 'test_static.dat', 'rb'))
    test_x_len = pickle.load(open(data_path + 'test_x_len.dat', 'rb'))
    test_mask_x = pickle.load(open(data_path + 'test_mask_x.dat', 'rb'))

    # ratio=0.05
    # train_len = len(train_x)
    # cut_len = int(train_len * ratio)
    # train_x = train_x[:cut_len]
    # train_y = train_y[:cut_len]
    # train_names = train_names[:cut_len]
    # train_static = train_static[:cut_len]
    # train_x_len = train_x_len[:cut_len]
    # train_mask_x = train_mask_x[:cut_len]

    # dev_len = len(dev_x)
    # cut_len = int(dev_len * ratio)
    # dev_x = dev_x[:cut_len]
    # dev_y = dev_y[:cut_len]
    # dev_names = dev_names[:cut_len]
    # dev_static = dev_static[:cut_len]
    # dev_x_len = dev_x_len[:cut_len]
    # dev_mask_x = dev_mask_x[:cut_len]

    # test_len = len(test_x)
    # cut_len = int(test_len * ratio)
    # test_x = test_x[:cut_len]
    # test_y = test_y[:cut_len]
    # test_names = test_names[:cut_len]
    # test_static = test_static[:cut_len]
    # test_x_len = test_x_len[:cut_len]
    # test_mask_x = test_mask_x[:cut_len]

    return (train_x, train_y, train_x_len,
            train_static), (dev_x, dev_y, dev_x_len,
                            dev_static), (test_x, test_y, test_x_len,
                                          test_static)


def load_challenge_frontfill():
    data_path = './data/Challenge/normalized/'

    all_x = pickle.load(open(data_path + 'x_front_fill.dat', 'rb'))
    all_y = pickle.load(open(data_path + 'y_front_fill.dat', 'rb'))
    all_names = pickle.load(open(data_path + 'name.dat', 'rb'))
    static = pickle.load(open(data_path + 'demo.dat', 'rb'))
    mask_x = pickle.load(open(data_path + 'mask_x.dat', 'rb'))
    mask_demo = pickle.load(open(data_path + 'mask_demo.dat', 'rb'))
    all_x_len = [len(i) for i in all_x]

    train_num = int(len(all_x) * 0.8) + 1
    dev_num = int(len(all_x) * 0.1) + 1
    test_num = int(len(all_x) * 0.1)
    assert (train_num + dev_num + test_num == len(all_x))

    train_x = []
    train_y = []
    train_names = []
    train_static = []
    train_x_len = []
    train_mask_x = []
    for idx in range(train_num):
        train_x.append(all_x[idx])
        train_y.append(int(all_y[idx][-1]))
        train_names.append(all_names[idx])
        train_static.append(static[idx])
        train_x_len.append(all_x_len[idx])
        train_mask_x.append(mask_x[idx])

    dev_x = []
    dev_y = []
    dev_names = []
    dev_static = []
    dev_x_len = []
    dev_mask_x = []
    for idx in range(train_num, train_num + dev_num):
        dev_x.append(all_x[idx])
        dev_y.append(int(all_y[idx][-1]))
        dev_names.append(all_names[idx])
        dev_static.append(static[idx])
        dev_x_len.append(all_x_len[idx])
        dev_mask_x.append(mask_x[idx])

    test_x = []
    test_y = []
    test_names = []
    test_static = []
    test_x_len = []
    test_mask_x = []
    for idx in range(train_num + dev_num, train_num + dev_num + test_num):
        test_x.append(all_x[idx])
        test_y.append(int(all_y[idx][-1]))
        test_names.append(all_names[idx])
        test_static.append(static[idx])
        test_x_len.append(all_x_len[idx])
        test_mask_x.append(mask_x[idx])

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)

    return (train_x, train_y, train_x_len,
            train_static), (dev_x, dev_y, dev_x_len,
                            dev_static), (test_x, test_y, test_x_len,
                                          test_static)


def load_challenge_zerofill_partial(ratio=0.1):
    data_path = './data/Challenge/normalized/'

    train_x = pickle.load(open(data_path + 'train_x.dat', 'rb'))
    train_y = pickle.load(open(data_path + 'train_y.dat', 'rb'))
    train_names = pickle.load(open(data_path + 'train_names.dat', 'rb'))
    train_static = pickle.load(open(data_path + 'train_static.dat', 'rb'))
    train_x_len = pickle.load(open(data_path + 'train_x_len.dat', 'rb'))
    train_mask_x = pickle.load(open(data_path + 'train_mask_x.dat', 'rb'))

    train_len = len(train_x)
    cut_len = int(train_len * ratio)
    train_x = train_x[:cut_len]
    train_y = train_y[:cut_len]
    train_names = train_names[:cut_len]
    train_static = train_static[:cut_len]
    train_x_len = train_x_len[:cut_len]
    train_mask_x = train_mask_x[:cut_len]
    # print(train_x_len)

    dev_x = pickle.load(open(data_path + 'dev_x.dat', 'rb'))
    dev_y = pickle.load(open(data_path + 'dev_y.dat', 'rb'))
    dev_names = pickle.load(open(data_path + 'dev_names.dat', 'rb'))
    dev_static = pickle.load(open(data_path + 'dev_static.dat', 'rb'))
    dev_x_len = pickle.load(open(data_path + 'dev_x_len.dat', 'rb'))
    dev_mask_x = pickle.load(open(data_path + 'dev_mask_x.dat', 'rb'))

    test_x = pickle.load(open(data_path + 'test_x.dat', 'rb'))
    test_y = pickle.load(open(data_path + 'test_y.dat', 'rb'))
    test_names = pickle.load(open(data_path + 'test_names.dat', 'rb'))
    test_static = pickle.load(open(data_path + 'test_static.dat', 'rb'))
    test_x_len = pickle.load(open(data_path + 'test_x_len.dat', 'rb'))
    test_mask_x = pickle.load(open(data_path + 'test_mask_x.dat', 'rb'))

    return (train_x, train_y, train_x_len,
            train_static), (dev_x, dev_y, dev_x_len,
                            dev_static), (test_x, test_y, test_x_len,
                                          test_static)


def load_challenge_new_full(time_gap = False):
    data_path = './data/Challenge/new/'

    all_x = pickle.load(open(data_path + 'x.dat', 'rb'))
    all_y = pickle.load(open(data_path + 'y.dat', 'rb'))
    static = pickle.load(open(data_path + 'demo.dat', 'rb'))
    all_x_mask = pickle.load(open(data_path + 'new_mask_x.dat', 'rb'))
    all_x_len = [len(i) for i in all_x]
    all_time = []
    if time_gap:
        for i in range(len(all_x_mask)):
            time_person = []
            for j in range(len(all_x_mask[i])):
                time_feature = []
                gap = 0
                for k in range(len(all_x_mask[i][j])):
                    time_feature.append(gap)
                    if all_x_mask[i][j][k] == 1:
                        gap = 1
                    else:
                        gap += 1
                time_person.append(time_feature)
            all_time.append(time_person)
    else:
        for i in range(len(all_x_len)):
            t0 = 1 / all_x_len[i]
            all_time.append([_*t0 for _ in range(all_x_len[i])])

    train_num = int(len(all_x) * 0.8) + 1
    dev_num = int(len(all_x) * 0.1) + 1
    test_num = int(len(all_x) * 0.1)
    assert (train_num + dev_num + test_num == len(all_x))

    train_x = []
    train_y = []
    train_static = []
    train_x_len = []
    train_x_mask = []
    train_times = []
    for idx in range(train_num):
        train_x.append(all_x[idx])
        train_y.append(int(all_y[idx][-1]))
        train_static.append(static[idx])
        train_x_len.append(all_x_len[idx])
        train_x_mask.append(all_x_mask[idx])
        train_times.append(all_time[idx])

    dev_x = []
    dev_y = []
    dev_static = []
    dev_x_len = []
    dev_x_mask = []
    dev_times = []
    for idx in range(train_num, train_num + dev_num):
        dev_x.append(all_x[idx])
        dev_y.append(int(all_y[idx][-1]))
        dev_static.append(static[idx])
        dev_x_len.append(all_x_len[idx])
        dev_x_mask.append(all_x_mask[idx])
        dev_times.append(all_time[idx])

    test_x = []
    test_y = []
    test_static = []
    test_x_len = []
    test_x_mask = []
    test_times = []
    for idx in range(train_num + dev_num, train_num + dev_num + test_num):
        test_x.append(all_x[idx])
        test_y.append(int(all_y[idx][-1]))
        test_static.append(static[idx])
        test_x_len.append(all_x_len[idx])
        test_x_mask.append(all_x_mask[idx])
        test_times.append(all_time[idx])

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)
    return (train_x, train_y, train_x_len, train_static, train_x_mask, train_times), (
        dev_x, dev_y, dev_x_len, dev_static, dev_x_mask, dev_times), (
        test_x, test_y, test_x_len, test_static, test_x_mask, test_times)


def load_challenge_new_full_file():
    return pickle.load(open('./data/Challenge/new/full.pkl', 'rb'))


def load_challenge_new_full_missing(time_gap = False, ratio = 1):
    data_path = './data/Challenge/new/'

    x = pickle.load(open(data_path + 'x.dat', 'rb'))
    y = pickle.load(open(data_path + 'y.dat', 'rb'))
    static = pickle.load(open(data_path + 'demo.dat', 'rb'))
    mask = pickle.load(open(data_path + 'new_mask_x.dat', 'rb'))
    x_len = [len(i) for i in x]
    y = [i[-1] for i in y]

    missing_x = deepcopy(x)
    missing_mask = deepcopy(mask)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            for k in range(len(mask[i][j])):
                if mask[i][j][k] != 0:
                    if random.random() > ratio:
                        missing_x[i][j][k] = 0
                        missing_mask[i][j][k] = 0
                    else:
                        mask[i][j][k] = 0

    time = []
    time_rev = []
    if time_gap:
        for i in range(len(missing_mask)):
            time_person = []
            time_person_rev = []
            for j in range(len(missing_mask[i])):
                time_feature = []
                time_feature_rev = []
                gap = 0
                gap_rev = 0
                length = len(missing_mask[i][j])
                for k in range(length):
                    time_feature.append(gap)
                    time_feature_rev.append(gap_rev)
                    if missing_mask[i][j][k] == 1:
                        gap = 1
                    else:
                        gap += 1
                    if missing_mask[i][j][length - k - 1] == 1:
                        gap_rev = 1
                    else:
                        gap_rev += 1
                time_person.append(time_feature)
                time_person_rev.append(time_feature_rev[::-1])
            time.append(time_person)
            time_rev.append(time_person_rev)
    else:
        for i in range(len(x_len)):
            t0 = 1 / x_len[i]
            time.append([_*t0 for _ in range(x_len[i])])

    train_num = int(len(x) * 0.8) + 1
    dev_num = int(len(x) * 0.1) + 1
    test_num = int(len(x) * 0.1)
    assert (train_num + dev_num + test_num == len(x))

    train_x = []
    train_y = []
    train_static = []
    train_x_len = []
    train_x_mask = []
    train_times = []
    train_times_rev = []
    train_missing_x = []
    train_missing_mask = []
    if time_gap:
        for idx in range(train_num):
            train_x.append(x[idx])
            train_y.append(int(y[idx]))
            train_static.append(static[idx])
            train_x_len.append(x_len[idx])
            train_x_mask.append(mask[idx])
            train_times.append(time[idx])
            train_times_rev.append(time_rev[idx])
            train_missing_x.append(missing_x[idx])
            train_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num):
            train_x.append(x[idx])
            train_y.append(int(y[idx]))
            train_static.append(static[idx])
            train_x_len.append(x_len[idx])
            train_x_mask.append(mask[idx])
            train_times.append(time[idx])
            train_missing_x.append(missing_x[idx])
            train_missing_mask.append(missing_mask[idx])

    dev_x = []
    dev_y = []
    dev_static = []
    dev_x_len = []
    dev_x_mask = []
    dev_times = []
    dev_times_rev = []
    dev_missing_x = []
    dev_missing_mask = []
    if time_gap:
        for idx in range(train_num, train_num + dev_num):
            dev_x.append(x[idx])
            dev_y.append(int(y[idx]))
            dev_static.append(static[idx])
            dev_x_len.append(x_len[idx])
            dev_x_mask.append(mask[idx])
            dev_times.append(time[idx])
            dev_times_rev.append(time_rev[idx])
            dev_missing_x.append(missing_x[idx])
            dev_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num, train_num + dev_num):
            dev_x.append(x[idx])
            dev_y.append(int(y[idx]))
            dev_static.append(static[idx])
            dev_x_len.append(x_len[idx])
            dev_x_mask.append(mask[idx])
            dev_times.append(time[idx])
            dev_missing_x.append(missing_x[idx])
            dev_missing_mask.append(missing_mask[idx])

    test_x = []
    test_y = []
    test_static = []
    test_x_len = []
    test_x_mask = []
    test_times = []
    test_times_rev = []
    test_missing_x = []
    test_missing_mask = []
    if time_gap:
        for idx in range(train_num + dev_num, train_num + dev_num + test_num):
            test_x.append(x[idx])
            test_y.append(int(y[idx]))
            test_static.append(static[idx])
            test_x_len.append(x_len[idx])
            test_x_mask.append(mask[idx])
            test_times.append(time[idx])
            test_times_rev.append(time_rev[idx])
            test_missing_x.append(missing_x[idx])
            test_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num + dev_num, train_num + dev_num + test_num):
            test_x.append(x[idx])
            test_y.append(int(y[idx]))
            test_static.append(static[idx])
            test_x_len.append(x_len[idx])
            test_x_mask.append(mask[idx])
            test_times.append(time[idx])
            test_missing_x.append(missing_x[idx])
            test_missing_mask.append(missing_mask[idx])

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)
    if time_gap:
        return (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_times_rev, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_static, dev_x_mask, dev_times, dev_times_rev, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_missing_x, test_missing_mask)
    else:
        return (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_static, dev_x_mask, dev_times, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_missing_x, test_missing_mask)


def load_challenge_2019(time_gap = True, ratio = 1):

    x, y, static, mask, name = pickle.load(open('./data/Challenge/data.pkl', 'rb'))
    x_len = [len(i) for i in x]

    missing_x = deepcopy(x)
    missing_mask = deepcopy(mask)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            for k in range(len(mask[i][j])):
                if mask[i][j][k] != 0:
                    if random.random() > ratio:
                        missing_x[i][j][k] = 0
                        missing_mask[i][j][k] = 0
                    else:
                        mask[i][j][k] = 0

    time = []
    time_rev = []
    if time_gap:
        for i in range(len(missing_mask)):
            time_person = []
            time_person_rev = []
            for j in range(len(missing_mask[i])):
                time_feature = []
                time_feature_rev = []
                gap = 0
                gap_rev = 0
                length = len(missing_mask[i][j])
                for k in range(length):
                    time_feature.append(gap)
                    time_feature_rev.append(gap_rev)
                    if missing_mask[i][j][k] == 1:
                        gap = 1
                    else:
                        gap += 1
                    if missing_mask[i][j][length - k - 1] == 1:
                        gap_rev = 1
                    else:
                        gap_rev += 1
                time_person.append(time_feature)
                time_person_rev.append(time_feature_rev[::-1])
            time.append(time_person)
            time_rev.append(time_person_rev)
    else:
        for i in range(len(x_len)):
            t0 = 1 / x_len[i]
            time.append([_*t0 for _ in range(x_len[i])])

    train_num = int(len(x) * 0.8)
    dev_num = int(len(x) * 0.1) + 1
    test_num = int(len(x) * 0.1)
    assert (train_num + dev_num + test_num == len(x))

    train_x = []
    train_y = []
    train_static = []
    train_x_len = []
    train_x_mask = []
    train_times = []
    train_times_rev = []
    train_missing_x = []
    train_missing_mask = []
    if time_gap:
        for idx in range(train_num):
            train_x.append(x[idx])
            train_y.append(int(y[idx]))
            train_static.append(static[idx])
            train_x_len.append(x_len[idx])
            train_x_mask.append(mask[idx])
            train_times.append(time[idx])
            train_times_rev.append(time_rev[idx])
            train_missing_x.append(missing_x[idx])
            train_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num):
            train_x.append(x[idx])
            train_y.append(int(y[idx]))
            train_static.append(static[idx])
            train_x_len.append(x_len[idx])
            train_x_mask.append(mask[idx])
            train_times.append(time[idx])
            train_missing_x.append(missing_x[idx])
            train_missing_mask.append(missing_mask[idx])

    dev_x = []
    dev_y = []
    dev_static = []
    dev_x_len = []
    dev_x_mask = []
    dev_times = []
    dev_times_rev = []
    dev_missing_x = []
    dev_missing_mask = []
    if time_gap:
        for idx in range(train_num, train_num + dev_num):
            dev_x.append(x[idx])
            dev_y.append(int(y[idx]))
            dev_static.append(static[idx])
            dev_x_len.append(x_len[idx])
            dev_x_mask.append(mask[idx])
            dev_times.append(time[idx])
            dev_times_rev.append(time_rev[idx])
            dev_missing_x.append(missing_x[idx])
            dev_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num, train_num + dev_num):
            dev_x.append(x[idx])
            dev_y.append(int(y[idx]))
            dev_static.append(static[idx])
            dev_x_len.append(x_len[idx])
            dev_x_mask.append(mask[idx])
            dev_times.append(time[idx])
            dev_missing_x.append(missing_x[idx])
            dev_missing_mask.append(missing_mask[idx])

    test_x = []
    test_y = []
    test_static = []
    test_x_len = []
    test_x_mask = []
    test_times = []
    test_times_rev = []
    test_missing_x = []
    test_missing_mask = []
    if time_gap:
        for idx in range(train_num + dev_num, train_num + dev_num + test_num):
            test_x.append(x[idx])
            test_y.append(int(y[idx]))
            test_static.append(static[idx])
            test_x_len.append(x_len[idx])
            test_x_mask.append(mask[idx])
            test_times.append(time[idx])
            test_times_rev.append(time_rev[idx])
            test_missing_x.append(missing_x[idx])
            test_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num + dev_num, train_num + dev_num + test_num):
            test_x.append(x[idx])
            test_y.append(int(y[idx]))
            test_static.append(static[idx])
            test_x_len.append(x_len[idx])
            test_x_mask.append(mask[idx])
            test_times.append(time[idx])
            test_missing_x.append(missing_x[idx])
            test_missing_mask.append(missing_mask[idx])

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)
    if time_gap:
        return (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_times_rev, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_static, dev_x_mask, dev_times, dev_times_rev, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_missing_x, test_missing_mask)
    else:
        return (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_static, dev_x_mask, dev_times, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_missing_x, test_missing_mask)