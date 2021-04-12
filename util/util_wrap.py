import gc
import math
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .timer import time2str


def train_test_indices(num_data, ratio_tests, seed=None):
    indices = np.arange(0, num_data)
    i_train, i_valid_tests = train_test_split(indices, test_size=ratio_tests, shuffle=True, random_state=seed)
    return i_train, i_valid_tests


def train_valid_test_split(num_data, ratio_valid, ratio_tests, seed=None):
    indices = np.arange(0, num_data)
    i_train, i_valid_tests, _, _ = train_test_split(indices, indices, test_size=ratio_valid + ratio_tests, shuffle=True, random_state=seed)
    if ratio_tests == 0:
        return i_train, i_valid_tests, []
    elif ratio_valid == 0:
        return i_train, [], i_valid_tests

    i_valid, i_tests, _, _ = train_test_split(i_valid_tests, i_valid_tests, test_size=ratio_tests / (ratio_valid + ratio_tests), shuffle=True, random_state=seed)
    return i_train, i_valid, i_tests


class EarlyStopper():
    # only Loss
    def __init__(self, patience=-1, verbose=0, delta=0):

        self.min_delta = 0
        self.patience = patience
        self.verbose = verbose
        self.waiting = 0

        self.best_state = None

        self.delta = delta

        # buf
        # the smaller, the better
        self.lowest_loss = math.inf

        # flag
        self.stop_flag = False
        self.lowest_signal = False

    def set(self, model: nn.Module, val_loss: float):
        val_loss = float(val_loss)

        self.lowest_signal = False

        if val_loss + self.delta < self.lowest_loss:
            self.best_state = deepcopy(model.state_dict())
            self.lowest_loss = val_loss
            self.lowest_signal = True
            self.waiting = 0
            if self.verbose > 0:
                print('stopper: lowest loss:', val_loss)
        else:
            self.waiting += 1
            if self.patience > 0 and self.waiting > self.patience:
                self.stop_flag = True

    def load_state(self, model):
        # load the best model
        model.load_state_dict(self.best_state)
        return model

    @property
    def stop(self):
        return self.stop_flag


class WithDummy:
    def __init__(self, x):
        self.x = x

    def __enter__(self):
        return self.x

    def __exit__(self, type, value, traceback):
        return


def fit(model, model_forward, x_train, x_valid, criterion_train, criterion_valid, additional_loss, n_epochs, optimizer, save_file_head=None, verbose=0, retain_graph=False):
    # fit(model, None, x_train, x_valid, criterion_train, criterion_valid, additional_loss, n_epochs, optimizer, save_file_head, verbose, retain_graph=False)

    logs = []
    stopper = EarlyStopper(patience=-1, verbose=0, delta=0)

    if 1 < verbose:
        print("train:", len(x_train[0]), " valid:", len(x_valid[0]))

    if model_forward is None:
        model_forward = model.forward

    if additional_loss is not None and additional_loss() is None:
        additional_loss = None

    start = time.time()
    model.train()

    tdm = tqdm if verbose == 1 else WithDummy
    with tdm(range(n_epochs)) as t:
        for epoch in t:
            gc.collect()
            output = model_forward(*x_train)
            loss = criterion_train(output)
            if additional_loss is not None:
                loss += additional_loss()
                add_running = additional_loss().item()
            else:
                add_running = 0
            loss_running = loss.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True if retain_graph else None)
            optimizer.step()

            # show result
            with torch.no_grad():
                model.eval()

                # valid loss
                output_valid = model_forward(*x_valid)
                loss_valid = criterion_valid(output_valid).item()

                if verbose == 1:
                    t.set_postfix(lowest=stopper.lowest_loss)

                stopper.set(model, loss_valid)

                if 1 < verbose or save_file_head is not None:

                    # train loss
                    output_train = model_forward(*x_train)
                    loss_train = criterion_train(output_train).item()

                    alog = [epoch, loss_running, loss_train, loss_valid, add_running]
                    logs.append(alog)

                    elapse = time.time() - start
                    estimated = elapse / (epoch + 1) * (n_epochs + 1)

                    txt = '%4d:' % epoch
                    txt += 'run_loss:%10.4f, ' % loss_running
                    txt += 'train_loss:%10.4f, ' % loss_train
                    txt += 'valid_loss:%10.4f, ' % loss_valid
                    if additional_loss is not None:
                        txt += 'add_loss:%10.4f, ' % add_running
                    txt += '%s' % time2str(elapse)
                    txt += '/%s' % time2str(estimated)
                    if stopper.lowest_signal:
                        txt += ' (lowest:%.4f)' % loss_valid
                    if 1 < verbose:
                        print(txt)
                if stopper.stop:  # or loss_valid < 1e-2:
                    break

            model.train()

    if 1 < verbose:
        print("load lowest:", stopper.lowest_loss)
    stopper.load_state(model)
    model.eval()

    if save_file_head is None:
        return

    # save log
    df = pd.DataFrame(logs)
    df.columns = ["epochs", 'running', "loss-train", "loss_valid", 'running_add']
    df.set_index(df.columns[0], inplace=True)
    df.to_csv(save_file_head + ".csv")
