import math
import numpy as np
import torch
import logging
import random


def pad_sents(sents, pad_token):
    sents_padded = []

    max_length = max([len(_) for _ in sents])
    for i in sents:
        padded = list(i) + [pad_token] * (max_length - len(i))
        sents_padded.append(np.array(padded))

    return np.array(sents_padded)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len,
                        device=length.device, dtype=length.dtype).expand(
                            len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def batch_iter(args, batch_size=256, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(args[0]) / batch_size)  # 向下取整
    index_array = list(range(len(args[0])))
    arg_len = len(args)

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size:(i + 1) *
                              batch_size]  # fetch out all the induces

        examples = []
        for idx in indices:
            e = [arg[idx] for arg in args]
            examples.append(e)

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        yield [[e[j] for e in examples] for j in range(arg_len)]


def get_loss(y_pred, y_true):
    loss = torch.nn.BCELoss(weight=None)
    return loss(y_pred, y_true)


def get_re_loss(y_pred, y_true):
    loss = torch.nn.MSELoss()
    return loss(y_pred, y_true)


def get_data_by_index(idxs, *args):
    ret = []
    for arg in args:
        ret.append([arg[i] for i in idxs])
    return ret


def get_n2n_data(x, y, x_len, *args):
    length = len(x)
    assert length == len(y)
    assert length == len(x_len)
    for arg in args:
        assert length == len(arg)
    arg_len = len(args)
    new_args = []
    for i in range(arg_len + 3):
        new_args.append([])
    for i in range(length):
        assert len(x[i]) == len(y[i])
        for arg in args:
            assert len(x[i]) == len(arg[i])
        for j in range(len(x[i])):
            new_args[0].append(x[i][:j + 1])
            new_args[1].append(y[i][j])
            new_args[2].append(j + 1)
            for k in range(arg_len):
                new_args[k + 3].append(args[k][i][:j + 1])
    return new_args


def split_train_valid_test(dataset, train_ratio=0.8, dev_ratio=0.1):
    """
    test_ratio is calculated by train_ratio and dev_ratio
    """
    all_num = len(dataset[0])
    train_num = int(all_num * train_ratio)
    dev_num = int(all_num * dev_ratio)
    test_num = all_num - train_num - dev_num

    train_set = []
    dev_set = []
    test_set = []
    for data in dataset:
        train_set.append(data[:train_num])
        dev_set.append(data[train_num:train_num + dev_num])
        test_set.append(data[train_num + dev_num:])

    return train_set, dev_set, test_set

def random_init(dataset, num_centers, device='cuda'):
    num_points = dataset.size(0)
    dimension = dataset.size(1)

    indices = torch.tensor(np.array(random.sample(
        range(num_points), k=num_centers)), dtype=torch.long)

    indices = indices.to(device)
    centers = torch.gather(
        dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

# Compute for each data point the closest center


def compute_codes(dataset, centers, device='cuda'):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes

# Compute new centers as means of the data points forming the clusters


def update_centers(dataset, codes, num_centers, device='cuda'):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension,
                          dtype=torch.float, device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(
        num_points, dtype=torch.float, device=device))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(
        num_centers, dtype=torch.float, device=device))
    centers /= cnt.view(-1, 1)
    return centers


def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        #         sys.stdout.write('.')
        #         sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            #             sys.stdout.write('\n')
            #             print('Converged in %d iterations' % num_iterations)
            break
        if num_iterations > 2000:
            logging.info('Not converged in %d iterations!' % num_iterations)
            break
        codes = new_codes
    return centers, codes
