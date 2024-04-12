import math
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
import logging
from scipy.optimize import linear_sum_assignment

from util import metrics
from util.log import Logging
from util.utils import batch_iter, pad_sents, get_loss


def log(x):
    return torch.log(x + 1e-8)


class Runner():
    def __init__(self,
                 args,
                 model=None,
                 optimizer=None,
                 lr_scheduler=None,
                 cal_loss=get_loss,
                 train_data=None,
                 dev_data=None,
                 test_data=None,
                 batch_size=256,
                 pad_token_x=None,
                 pad_token_static=None,
                 kfold=False,
                 save_model=True,
                 use_static=False,
                 device='cuda'):
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.prototype_loss_ratio = args.prototype_loss_ratio
        self.push_start = 10
        self.push_epochs = args.push_epochs
        self.cal_loss = cal_loss
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.pad_token_x = pad_token_x
        self.pad_token_static = pad_token_static
        self.kfold = kfold
        self.save_model = save_model
        self.save_dir = args.save_dir
        self.device = device
        self.history = []
        self.global_best = None
        self.fixed_length = args.fixed_length
        self.use_static = use_static
        self.logging = Logging(self.epochs * 10 if self.kfold else self.epochs)
        if self.kfold:
            self.global_best = 0
            self.total_train_loss = []
            self.total_valid_loss = []
        self.proto_index = -1

    def reset(self):
        self.history = []
        if self.kfold:
            self.global_best = 0
            self.total_train_loss = []
            self.total_valid_loss = []

    def _tensorize(self, data):
        lens = None
        static = None
        if self.dataset == 'esrd':
            if self.use_static:
                x, y, lens, static = data
            else:
                x, y, lens = data
        elif self.dataset == 'challenge' or self.dataset == 'c12':
            if self.use_static:
                x, y, lens, static = data
            else:
                x, y, lens = data
        elif self.dataset == 'mimic':
            if self.use_static:
                x, y, static = data
            else:
                x, y = data
        else:
            raise ValueError("Unsupported data!")
        y = torch.tensor(y, dtype=torch.float32).to(
            self.device, non_blocking=True)
        if self.fixed_length:
            x = torch.tensor(x, dtype=torch.float32).to(
                self.device, non_blocking=True)
            if self.use_static:
                static = torch.tensor(static, dtype=torch.float32).to(
                    self.device, non_blocking=True)
        else:
            x = torch.tensor(pad_sents(x, self.pad_token_x), dtype=torch.float32).to(
                self.device, non_blocking=True)
            lens = torch.tensor(lens, dtype=torch.int32)
            if self.use_static:
                static = torch.tensor(pad_sents(static, self.pad_token_static),
                                      dtype=torch.float32).to(self.device, non_blocking=True)
                # if self.dataset == 'challenge':
                #     static = static[:, -1]
        return x, y, lens, static

    def _compute_loss(self, opt, label, distance, h_t):
        bce_loss = self.cal_loss(opt, label.unsqueeze(-1))
        return bce_loss

    def _train(self, each_epoch):
        epoch_loss = []
        self.model.train()
        for step, batch_train in enumerate(
                batch_iter(self.train_data,
                           batch_size=self.batch_size,
                           shuffle=True)):
            x, y, lens, static = self._tensorize(batch_train)
            opt, distance, ht = self.model(x, lens, static)
            loss = self._compute_loss(opt, y, distance, ht)
            self.model.Attention1.requires_grad_(False)
            self.model.output.requires_grad_(False)
            opt = self.model(train_proto=True)[0]
            loss_p = 0
            for i in range(self.model.num_prototypes):
                for j in range(i + 1, self.model.num_prototypes):
                    loss_p += opt[i] * log(opt[j]) + (1. - opt[i]) * log(1. - opt[j])
            loss_p /= (self.model.num_prototypes) * (self.model.num_prototypes - 1) / 2
            loss_d = torch.clamp(70 / math.sqrt(self.model.num_prototypes) - torch.pdist(self.model.prototype_vectors.reshape(self.model.num_prototypes, -1)), 0).mean()
            loss = loss + self.prototype_loss_ratio * loss_p + self.prototype_loss_ratio * loss_d
            self.model.Attention1.requires_grad_(True)
            self.model.output.requires_grad_(True)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            epoch_loss.append(loss.cpu().detach().numpy())

            if step % 40 == 0 and step != 0:
                self.logging.log_batch(
                    each_epoch, step, loss.cpu().detach().numpy())
        return np.mean(epoch_loss)

    def _test(self, test_data):
        valid_loss = []
        y_true_flatten = []
        y_pred_flatten = []
        distance = None
        self.model.eval()
        with torch.no_grad():
            for batch_test in batch_iter(test_data,
                                         batch_size=self.batch_size,
                                         shuffle=True):
                x, y, lens, static = self._tensorize(batch_test)
                opt, distance, ht = self.model(x, lens, static)
                loss = self._compute_loss(opt, y, distance, ht)
                valid_loss.append(loss.cpu().detach().numpy())
                y_pred_flatten += list(opt.cpu().detach().numpy().flatten())
                y_true_flatten += list(y.cpu().numpy().flatten())
            valid_loss = np.mean(valid_loss)
        return valid_loss, y_pred_flatten, y_true_flatten

    def train(self,
              model=None,
              optimizer=None,
              lr_scheduler=None,
              train_data=None,
              test_data=None,
              fold_count=None,
              print_interval=1):

        train_loss = []
        valid_loss = []
        best_auroc = 0
        if self.kfold:
            if not fold_count or not model or not optimizer or not train_data or not test_data:
                raise ValueError("kfold is specified and missing parameters!")
            self.model = model
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.train_data = train_data[0:-1]
            self.test_data = test_data

        hs = []
        for batch_train in batch_iter(self.train_data, batch_size=self.batch_size):
            x, y, lens, static = self._tensorize(batch_train)
            with torch.no_grad():
                hs.append(self.model.push_forward(x, lens, static))
        hs = torch.cat((hs), dim=0)
        kmeans = MiniBatchKMeans(n_clusters=self.model.num_prototypes)
        kmeans.fit(hs.reshape(hs.shape[0], -1).numpy())
        cluster_center = kmeans.cluster_centers_
        prototype_update = np.reshape(
            cluster_center, self.model.prototype_vectors.shape)
        self.model.prototype_vectors.data.copy_(
            torch.tensor(prototype_update, dtype=torch.float32).cuda())

        for each_epoch in range(self.epochs):
            epoch_train_loss = self._train(each_epoch)
            train_loss.append(epoch_train_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Validation
            if self.dev_data:
                epoch_valid_loss, y_pred, y_true = self._test(
                    self.dev_data)
            else:
                epoch_valid_loss, y_pred, y_true = self._test(
                    self.test_data)
            valid_loss.append(epoch_valid_loss)
            ret = metrics.print_metrics_binary(y_true, y_pred, verbose=0)
            self.history.append(ret)

            if self.kfold:
                self.logging.log_epoch(each_epoch, epoch_train_loss, epoch_valid_loss,
                                       ret['auroc'], fold_count)
            else:
                self.logging.log_epoch(each_epoch, epoch_train_loss, epoch_valid_loss,
                                       ret['auroc'])
                if each_epoch % print_interval == 0:
                    metrics.print_metrics_binary(y_true, y_pred)

            cur_auroc = ret['auroc']
            if cur_auroc > best_auroc:
                best_auroc = cur_auroc
                best_auprc = ret['auprc']
                best_minpse = ret['minpse']
                best_acc = ret['acc']
                best_f1_score = ret['f1_score']
                if self.save_model:
                    state = {
                        'net': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': each_epoch
                    }
                    if fold_count:
                        torch.save(state, os.path.join(
                            self.save_dir, 'fold10-' + str(fold_count) + '.pth'))
                    else:
                        logging.info(
                            '------------ Save best model - AUROC: %.4f ------------'
                            % cur_auroc)
                        torch.save(state, os.path.join(
                            self.save_dir, 'checkpoint.pth'))

                if self.global_best is not None and cur_auroc > self.global_best:
                    self.global_best = cur_auroc
                    if self.save_model:
                        state = {
                            'net': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': each_epoch,
                            'fold': fold_count,
                            'proto_index': self.proto_index
                        }
                        torch.save(state, os.path.join(
                            self.save_dir, 'checkpoint.pth'))
                        logging.info(
                            '------------ Save best model - AUROC: %.4f ------------'
                            % cur_auroc)

            if self.kfold:
                self.total_train_loss.append(train_loss)
                self.total_valid_loss.append(valid_loss)

            if each_epoch >= self.push_start and each_epoch % 1 == 0 and each_epoch not in self.push_epochs:
                self.model.eval()
                hs = []
                for push_iter, batch_train in enumerate(
                        batch_iter(self.train_data, batch_size=self.batch_size)):
                    x, y, lens, static = self._tensorize(batch_train)
                    with torch.no_grad():
                        hs.append(self.model.push_forward(x, lens, static))
                hs = torch.cat((hs), dim=0)
                distance = torch.norm(self.model.prototype_vectors.detach().cpu().reshape(
                    self.model.num_prototypes, -1).unsqueeze(1) - hs.reshape(
                    hs.shape[0], -1).unsqueeze(0), p=2, dim=-1)
                idx = distance.argmin(dim=1)
                self.model.prototype_vectors.data.copy_(hs[idx].cuda())

            if each_epoch >= self.push_start and each_epoch in self.push_epochs:
                self.model.init_mode = False
                self.push_prototypes()
                self.model.freeze()
                last_valid_loss = 100
                for i in range(5):
                    epoch_train_loss = self._train(i)
                    if self.dev_data:
                        epoch_valid_loss, y_pred, y_true = self._test(
                            self.dev_data)
                    else:
                        epoch_valid_loss, y_pred, y_true = self._test(
                            self.test_data)
                    ret = metrics.print_metrics_binary(y_true,
                                                       y_pred,
                                                       verbose=0)
                    if self.kfold:
                        logging.info(
                            "Fold %d, Epoch %d: Loss = %.4f Valid Loss = %.4f roc = %.4f"
                            % (fold_count, i, epoch_train_loss,
                               epoch_valid_loss, ret['auroc']))
                    else:
                        logging.info(
                            "Epoch %d: Loss = %.4f Valid Loss = %.4f roc = %.4f"
                            % (i, epoch_train_loss, epoch_valid_loss,
                               ret['auroc']))
                    if epoch_valid_loss > last_valid_loss:
                        break
                    last_valid_loss = epoch_valid_loss

                self.model.unfreeze()

        if self.dev_data:
            best_auroc, best_auprc, best_minpse, best_acc, best_f1_score = self.test()

        return best_auroc, best_auprc, best_minpse, best_acc, best_f1_score

    def test(self,
             model=None,
             optimizer=None,
             test_data=None,
             fold_count=None):

        valid_loss = []
        if self.kfold:
            if not fold_count or not model or not optimizer or not test_data:
                raise ValueError("kfold is specified and missing parameters!")
            self.model = model
            self.optimizer = optimizer
            self.test_data = test_data
            checkpoint = torch.load(os.path.join(
                self.save_dir, 'fold10-' + str(fold_count) + '.pth'))
        else:
            checkpoint = torch.load(os.path.join(
                self.save_dir, 'checkpoint.pth'))
        save_epoch = checkpoint['epoch']
        print("last saved model is in epoch {}".format(save_epoch))
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.eval()
        epoch_valid_loss, y_pred, y_true = self._test(self.test_data)
        valid_loss.append(epoch_valid_loss)
        ret = metrics.print_metrics_binary(y_true, y_pred, verbose=0)
        self.history.append(ret)

        if self.kfold:
            print('Fold %d, Valid loss = %.4f roc = %.4f' %
                  (fold_count, valid_loss[-1], ret['auroc']),
                  flush=True)
        else:
            print('Valid loss = %.4f roc = %.4f' %
                  (valid_loss[-1], ret['auroc']),
                  flush=True)
        metrics.print_metrics_binary(y_true, y_pred)

        if self.kfold:
            self.total_valid_loss.append(valid_loss)

        return ret['auroc'], ret['auprc'], ret['minpse'], ret['acc'], ret[
            'f1_score']

    def push_prototypes(self):
        self.model.eval()
        prototype_shape = self.model.prototype_shape

        ys = []
        hs = []
        for push_iter, batch_train in enumerate(
                batch_iter(self.train_data, batch_size=self.batch_size)):
            x, y, lens, static = self._tensorize(batch_train)
            with torch.no_grad():
                hs.append(self.model.push_forward(x, lens, static))
        hs = torch.cat((hs), dim=0)
        self.update_prototypes_on_batch(hs)

        prototype_update = np.reshape(self.global_fmap_patches,
                                      tuple(prototype_shape))
        self.model.prototype_vectors.data.copy_(
            torch.tensor(prototype_update, dtype=torch.float32).cuda())

    def update_prototypes_on_batch(self, h_t):
        h_t = h_t.detach().numpy()
        h_ti = h_t.reshape((h_t.shape[0], -1))
        kmeans = MiniBatchKMeans(n_clusters=self.model.num_prototypes)
        kmeans.fit(h_ti)
        cluster_center = kmeans.cluster_centers_

        # find nearest sample
        distance = np.linalg.norm(np.expand_dims(
            cluster_center, 1) - np.expand_dims(h_ti, 0), axis=-1)
        index = np.argmin(distance, axis=-1)
        self.proto_index = index
        cluster_center = h_ti[index]

        distance = np.linalg.norm(self.model.prototype_vectors.detach().cpu().reshape(
            (self.model.num_prototypes, -1)).unsqueeze(1).numpy() - np.expand_dims(cluster_center, 0), ord=2, axis=-1)
        rank = linear_sum_assignment(distance)[1]
        cluster_center = cluster_center[rank]

        if self.use_static:
            cluster_center = cluster_center.reshape(
                (self.model.num_prototypes, self.model.input_dim + 1,
                 self.model.hidden_dim))
        else:
            cluster_center = cluster_center.reshape(
                (self.model.num_prototypes, self.model.input_dim,
                 self.model.hidden_dim))
        self.global_fmap_patches = cluster_center
