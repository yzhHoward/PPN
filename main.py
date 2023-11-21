import numpy as np
import os
import random
from sklearn.model_selection import StratifiedKFold
import torch
import argparse
import logging
import json

from util.log import init_logging
from util.runner import Runner
from util.utils import get_data_by_index, split_train_valid_test, get_n2n_data
from data.ckd import load_ckd_mortality_few_visit, load_ckd_mortality_sparse, load_ckd_mortality_zerofill
from data.challenge import load_challenge_zerofill, load_challenge_new, load_challenge_new_few_visit, load_challenge_new_sparse
from data.challenge2012 import load_challenge_2012_file
from data.mimic_iii import mimic_init, load_mimic_mortality, load_mimic_mortality_file, load_mimic_mortality_few_visit, load_mimic_mortality_sparse
from model.promanet import Promanet

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')

auroc = []
auprc = []
minpse = []
acc = []
f1_score = []


def vanilla_run(args, data):
    train_data, dev_data, test_data = data
    if args.fixed_length:
        pad_token_x = None
        pad_token_static = None
    else:
        pad_token_x = np.zeros(args.x_dim)
        pad_token_static = np.zeros(args.demo_dim)
    model = Promanet(input_dim=args.x_dim,
                     hidden_dim=args.hidden_dim,
                     output_dim=args.output_dim,
                     demo_dim=args.demo_dim,
                     drop_rate=args.drop_rate,
                     num_prototypes=args.num_prototypes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    runner = Runner(args,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=None,
                    train_data=train_data,
                    dev_data=dev_data,
                    test_data=test_data,
                    pad_token_x=pad_token_x,
                    pad_token_static=pad_token_static,
                    batch_size=args.batch_size,
                    kfold=args.kfold,
                    save_model=args.save_model,
                    use_static=True,
                    device=device)

    if not args.test:
        best_auroc, best_auprc, best_minpse, best_acc, best_f1_score = runner.train(
            print_interval=10)
    else:
        best_auroc, best_auprc, best_minpse, best_acc, best_f1_score = runner.test()
    auroc.append(best_auroc)
    auprc.append(best_auprc)
    minpse.append(best_minpse)
    acc.append(best_acc)
    f1_score.append(best_f1_score)


def kfold_run(args, data):
    x, y, lens, static, name, survival_dict = data
    pad_token_x = np.zeros(args.x_dim)
    pad_token_static = np.zeros(args.demo_dim)
    kfold = StratifiedKFold(n_splits=10,
                            shuffle=True,
                            random_state=RANDOM_SEED)
    runner = Runner(args,
                    pad_token_x=pad_token_x,
                    pad_token_static=pad_token_static,
                    batch_size=args.batch_size,
                    kfold=args.kfold,
                    save_model=args.save_model,
                    use_static=True,
                    device=device)
    fold_count = 0

    for train_set, test_set in kfold.split(x, y):
        model = Promanet(input_dim=args.x_dim,
                         hidden_dim=args.hidden_dim,
                         output_dim=args.output_dim,
                         demo_dim=args.demo_dim,
                         drop_rate=args.drop_rate,
                         num_prototypes=args.num_prototypes).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                     lr=args.lr)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=lr_step_func)

        train_data = get_data_by_index(train_set, x, y, lens, static, name)
        test_data = get_data_by_index(test_set, x, y, lens, static)

        fold_count += 1
        if not args.test:
            best_auroc, best_auprc, best_minpse, best_acc, best_f1_score = runner.train(
                model=model,
                optimizer=optimizer,
                lr_scheduler=None,
                train_data=train_data,
                test_data=test_data,
                fold_count=fold_count,
                print_interval=10)
        else:
            best_auroc, best_auprc, best_minpse, best_acc, best_f1_score = runner.test(
                model=model,
                optimizer=optimizer,
                test_data=test_data,
                fold_count=fold_count)
        auroc.append(best_auroc)
        auprc.append(best_auprc)
        minpse.append(best_minpse)
        acc.append(best_acc)
        f1_score.append(best_f1_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--dataset',
                        choices=['esrd', 'challenge', 'mimic', 'c12'],
                        default='esrd')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--prototype_loss_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--drop_rate', type=float, default=0.4)
    parser.add_argument('--num_prototypes', type=int, default=6)
    parser.add_argument('--push_epochs', type=list, default=[10, 30, 50])
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./export/ckd/6-1-seed42/',
        # default='./export/c12/16-seed00',
        # default='./export/challenge/16-2-seed0',
        # default='./export/mimic/8-2-seed42',
    )
    args = parser.parse_args()
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)  # numpy
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)  # cpu
    torch.cuda.manual_seed(RANDOM_SEED)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    if args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_root = logging.getLogger()
    init_logging(
        log_root, args.save_dir if not args.test and args.save_model else None)
    logging.info(json.dumps(vars(args), indent=4))
    if args.dataset == 'esrd':
        args.kfold = True
        data = load_ckd_mortality_zerofill(
        )  # x, y, lens, static, name, survival_dict
        # data = load_ckd_mortality_few_visit(0.75)
        # data = load_ckd_mortality_sparse(0.75)
        args.x_dim = 17
        args.demo_dim = 4
        args.fixed_length = False
    elif args.dataset == 'c12':
        args.kfold = False
        data = load_challenge_2012_file()  # train_data, dev_data, test_data
        # data = load_challenge_new_few_visit(0.7)
        # data = load_challenge_new_sparse(0.7)
        args.x_dim = 37
        args.demo_dim = 4
        args.max_len = 48
        args.fixed_length = False
    elif args.dataset == 'challenge':
        args.kfold = False
        # data = load_challenge_zerofill()  # train_data, dev_data, test_data
        data = load_challenge_new()  # train_data, dev_data, test_data
        # data = load_challenge_new_few_visit(0.25)
        # data = load_challenge_new_sparse(0.75)
        args.x_dim = 34
        args.demo_dim = 5
        args.fixed_length = False
    elif args.dataset == 'mimic':
        args.kfold = False
        # mimic_init()
        # if not args.test:
        #     data = load_mimic_mortality()  # train_data, dev_data, test_data
        # else:
        #     data = (None, None, load_mimic_mortality_test())
        data = load_mimic_mortality_file()
        # data = load_mimic_mortality_few_visit(0.8)
        # data = load_mimic_mortality_sparse(0.8)
        args.x_dim = 76
        args.demo_dim = 12
        args.fixed_length = True
    else:
        raise ValueError('Unsupport Dataset!')
    print('Dataset Loaded.')
    if args.kfold:
        kfold_run(args, data)
    else:
        vanilla_run(args, data)

    logging.info('auroc %.4f(%.4f)' % (np.mean(auroc), np.std(auroc)))
    logging.info('auprc %.4f(%.4f)' % (np.mean(auprc), np.std(auprc)))
    logging.info('minpse %.4f(%.4f)' % (np.mean(minpse), np.std(minpse)))
    logging.info('acc %.4f(%.4f)' % (np.mean(acc), np.std(acc)))
    logging.info('f1_score %.4f(%.4f)' % (np.mean(f1_score), np.std(f1_score)))
