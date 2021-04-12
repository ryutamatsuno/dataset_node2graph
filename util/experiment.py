import gc
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .timer import time2str

optuna.logging.set_verbosity(optuna.logging.WARNING)



class PrintTrial():
    def __init__(self):
        super().__init__()

    def suggest_loguniform(self, label, x, y):
        return '[' + "%.2e" % x + " -- " + "%.2e" % y + ')'

    def suggest_uniform(self, label, x, y):
        return '[' + str(x) + " -- " + str(y) + ')'

    def suggest_categorical(self, label: str, xs):
        return str(xs)

    def suggest_int(self, label, low, high):
        return '[' + str(low) + ',' + str(high) + ']'


def hs_opt_general(wrap, xs, ys, params, n_cv=5, n_try=50, save_head=None, njob=-1):
    print('\nHyperparameter search by Optuna')
    start = time.time()

    def objwrap(pbar):
        def objective(trial):
            search_param = params(trial)

            # printing for log
            if isinstance(trial, PrintTrial):
                base_param = wrap.get_params()
                base_param.update(search_param)
                print("search_params:\n{")
                for key in base_param:
                    print("  '" + (key + "'").ljust(11, " ") + ":" + str(base_param[key]) + ",")
                print("}")
                return None

            wrap.set_params(**search_param)
            # proper cv
            if isinstance(n_cv, int):
                if ys is not None:
                    folder = StratifiedKFold(n_splits=n_cv, shuffle=True)
                else:
                    folder = KFold(n_splits=n_cv, shuffle=True)
            else:
                folder = n_cv
            scores = cross_val_score(wrap, xs, ys, cv=folder, n_jobs=njob, verbose=0)
            # score : higher better
            # optuna: lower better
            scores = np.array(scores)
            res = np.average(scores)
            pbar.update()
            pbar.set_postfix(score=res)
            return -res

        return objective

    # print searching space
    (objwrap(None))(PrintTrial())

    study = optuna.create_study()
    with tqdm(total=n_try) as pbar:
        study.optimize(objwrap(pbar), n_trials=n_try)

    if save_head is not None:
        hist_df = study.trials_dataframe()
        hist_df.to_csv(save_head + '_opt.csv')

    print('\n')
    print('Search Result:', study.best_params)
    print('BestScore:', - study.best_value)
    elapsed = time.time() - start
    print('all done :', time2str(elapsed), ' average:', time2str(elapsed / n_try))

    wrap.set_params(**study.best_params)
    best_param = wrap.get_params()

    print('BestParam :', best_param)
    return best_param


def hs_grid_general(wrap, xs, ys, search_params, n_cv=5, save_head=None, njob=-1, verbose=0):
    if verbose > 0:
        print('\nHyperparameter search by grid search')
    start = time.time()

    if type(search_params) is list:
        common_searched_keys = None
        for x in search_params:
            if common_searched_keys is None:
                common_searched_keys = set(x.keys())
                continue
            common_searched_keys = common_searched_keys.intersection(set(x.keys()))
        # ordered keys
        key_list = [x for x in search_params[0].keys() if x in common_searched_keys]
        for x in search_params:
            for y in x.keys():
                if y in key_list:
                    continue
                key_list.append(y)
    else:
        common_searched_keys = search_params.keys()
        key_list = search_params.keys()

    # print base params
    base_param = wrap.get_params()
    if verbose > 0:
        print("base params:{")
        for key in base_param:
            if not key in common_searched_keys:
                print("  '" + (key + "'").ljust(11) + ":" + str(base_param[key]) + ",")
        print("}")

    def show(params):
        for key in params:
            print("  '" + (key + "'").ljust(11) + ":" + str(params[key]) + ",")

    if verbose > 0:
        if type(search_params) is list:
            print("search params:{")
            for i, params in enumerate(search_params):
                # print("{")
                show(params)
                if i < len(search_params) - 1:
                    print('}, {')
                else:
                    print('}')
        else:
            print("search params:{")
            show(search_params)
            print("}")

    if verbose > 0:
        print('searching...')
    # clf = GridSearchCV(wrap, search_params, cv=n_cv, verbose=0, n_jobs=njob, refit=False)
    clf = GridSearchCV(wrap, search_params, cv=n_cv, verbose=verbose - 1 if verbose > 1 else False, n_jobs=njob, refit=False)
    clf.fit(xs, ys)

    if verbose > 0:
        print('grid search finished:', time2str(time.time() - start))

    # printing
    l = len(clf.cv_results_['params'])
    idx = np.argsort([clf.cv_results_['rank_test_score'][i] for i in range(l)])

    # formating
    pls = {}
    if type(search_params) is list:
        for zzz in search_params:
            for p in zzz:
                l = max([len(str(x)) for x in zzz[p]])
                if p in pls:
                    pls[p] = max(l, pls[p])
                else:
                    pls[p] = l
    else:
        for p in search_params:
            l = max([len(str(x)) for x in search_params[p]])
            pls[p] = l

    for i in idx:
        params = clf.cv_results_['params'][i]
        mean_test_score = clf.cv_results_['mean_test_score'][i]
        std_test_score = clf.cv_results_['std_test_score'][i]
        rank_test_score = clf.cv_results_['rank_test_score'][i]

        param_txt = "{ "

        # for p in params:
        for p in key_list:
            if not p in params:
                param_txt += ''.ljust(len(p) + 2 + pls[p] + 1 + 1)
                continue
            param_txt += p + "= " + (str(params[p]) + ",").ljust(pls[p] + 1) + ' '
        param_txt += '}'

        if verbose > 0:
            print("rank:{:3d} {:.3f} (+- {:.3f}) for {:s}".format(rank_test_score, mean_test_score, std_test_score, param_txt))
        # print("rank:{:3d} {:.3f} (+- {:.3f}) for {}".format(rank_test_score, mean_test_score, std_test_score, params))

    if save_head is not None:
        gs_result = pd.DataFrame.from_dict(clf.cv_results_)
        gs_result.to_csv(save_head + '_grid.csv')

    if verbose > 0:
        # print('\n')
        print('BestParam:', clf.best_params_)
        print('BestScore:', clf.best_score_)
        print('all done:', time2str(time.time() - start))

    # exit()
    return clf.best_params_


def parameter_sensitivity(wrap, xs, ys, search_params, score_func=None, n_cv=5, save_head=None, njob=-1):
    start = time.time()

    assert save_head is not None, "please set save_head"

    base_param = wrap.get_params()
    print("base params:\n{")
    for key in base_param:
        print("  '" + (key + "'").ljust(11, " ") + ":" + str(base_param[key]) + ",")
    print("}")

    if ys is None:
        ys = np.ones(len(xs), dtype=np.int) * -1  # [-1 for _ in range(len(xs))]

    print('\nParameter sensitivity check\n')

    print("search_params:\n{")
    for key in search_params:
        print("  '" + (key + "'").ljust(11, " ") + ":" + str(search_params[key]) + ",")
    print("}")

    # if score_func is not None:
    #     score_func = make_scorer(score_func, greater_is_better=True)

    scoring = lambda: None if score_func is None else score_func

    print()
    for key in search_params:
        st1 = time.time()
        vals = search_params[key]
        print("Sensitivity to", key, ":", vals)
        l = len(vals)

        if l == 1:
            continue
        clf = GridSearchCV(wrap, {key: vals}, scoring=scoring(), cv=n_cv, verbose=0, n_jobs=njob, refit=False)
        clf.fit(xs, ys)

        scores = []

        print('Result of', key, ":")
        for i in range(l):
            param = clf.cv_results_['params'][i][key]
            mean_test_score = clf.cv_results_['mean_test_score'][i]
            std_test_score = clf.cv_results_['std_test_score'][i]
            scores.append("$ %.4f \pm %.4f $" % (mean_test_score, std_test_score))
            print("%8.4f" % param, scores[-1])
            assert param == vals[i]

        # save df
        df = pd.DataFrame()
        df['params'] = pd.Series(vals)
        df['scores'] = pd.Series(scores)
        df.set_index('params', inplace=True)
        filepath = save_head + "_ps_" + key + ".csv"
        df.to_csv(filepath, header=True, index=True)
        print("time:", time2str(time.time() - st1))
        print('plot ' + filepath, '--std --labelx', key)
        print()

    print('Finish parameter sensitivity check')
    print('all done:', time2str(time.time() - start))


def cv_general_eval(wrap, xs, ys, n_cv=10, n_time=10, save_head=None, njob=-1):
    start = time.time()

    # result is average of all of the results
    w_evals = []
    for j in tqdm(range(n_time)):
        if isinstance(n_cv, int):
            if ys is None:
                splits = KFold(n_splits=n_cv, shuffle=True, random_state=100 + j).split(xs)
            else:
                splits = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=100 + j).split(xs, ys)
        else:
            splits = n_cv.split(xs, ys)

        def process(wrap, xs, ys, i_train, i_tests):
            if isinstance(xs, list):
                x_train = [xs[i] for i in i_train]
                x_tests = [xs[i] for i in i_tests]
            else:
                x_train = xs[i_train]
                x_tests = xs[i_tests]

            if ys is None:
                y_train = None
                y_tests = None
            else:
                if isinstance(ys, list):
                    y_train = [ys[i] for i in i_train]
                    y_tests = [ys[i] for i in i_tests]
                else:
                    y_train = ys[i_train]
                    y_tests = ys[i_tests]

            estimator = clone(wrap)
            estimator.fit(x_train, y_train)
            eval = estimator.evaluation(x_tests, y_tests, verbose=0)
            gc.collect()
            return eval

        if njob > 1:
            evals = Parallel(n_jobs=njob)(delayed(process)(wrap, xs, ys, i_train, i_tests) for i_train, i_tests in splits)
        else:
            evals = []
            for i_train, i_tests in splits:
                eval = process(wrap, xs, ys, i_train, i_tests)
                evals.append(eval)

        w_evals.extend(evals)

    evals = w_evals

    keys = list(evals[0].keys())

    # scores for each keys
    resutls = concatinating_evals(evals)
    aves = []
    stds = []
    for key in keys:
        scores = resutls[key]

        ave = float(np.average(scores))
        std = float(np.std(scores))

        aves.append(ave)
        stds.append(std)

        # print for latex
        print()
        print('[', key, ']')
        print('$ %.3f \pm %.3f $' % (round(ave, 3), round(std, 3)))
        print('$ %f \pm %f $' % (ave, std))
        print(scores)

    # save log
    if save_head is not None:
        fpath = save_head + '_av_eval.csv'
        with open(fpath, 'w') as f:
            # head
            f.write('i,' + ','.join(keys) + '\n')
            f.write('ave,' + ','.join([str(ave) for ave in aves]) + '\n')
            f.write('std,' + ','.join([str(ave) for ave in stds]) + '\n')
            for i in range(len(evals)):
                f.write('%3d,' % i + ','.join([str(evals[i][key]) for key in keys]) + '\n')
    print()
    print('all done:', time2str(time.time() - start))

    return resutls


def run_general(wrap, x, y, test_size=0.2, verbose=1, save_head=None):
    start = time.time()
    # single run
    if y is None:
        x_train, x_tests = train_test_split(x, test_size=test_size, shuffle=True, random_state=None)
        y_train = None
        y_tests = None
    else:
        x_train, x_tests, y_train, y_tests = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=None)
    if save_head is not None:
        save_head = save_head + '_run'
    wrap.fit(x_train, y_train, verbose=verbose, save_file_head=save_head)
    eval = wrap.evaluation(x_tests, y_tests, verbose=2)

    print('run done:', time2str(time.time() - start))
    return eval


# eval

def runtime_general(wrap, xs, ys, test_size=0.2, niter=100):
    start = time.time()

    def process(wrap, xs, ys, test_size, random):
        if ys is None:
            x_train, x_tests = train_test_split(xs, test_size=test_size, shuffle=True, random_state=100 + random)
            y_train = None
            y_tests = None
        else:
            x_train, x_tests, y_train, y_tests = train_test_split(xs, ys, test_size=test_size, shuffle=100 + random, random_state=None)
        estimator = clone(wrap)
        s = time.time()
        estimator.fit(x_train, y_train)
        return time.time() - s

    # serial (mainly to computes time)
    times = []
    for i in tqdm(range(niter)):
        t = process(wrap, xs, ys, test_size, i)
        times.append(t)
    print('running time:', time2str(float(np.average(times))))
    print('running time:', '$ %.3f \pm %.3f $' % (round(float(np.average(times)), 3), round(float(np.std(times)), 3)))
    print('running time:', '$ %f \pm %f $' % (float(np.average(times)), float(np.std(times))))
    print('all done:', time2str(time.time() - start))
    return


def av_general_eval(wrap, xs, ys, test_size=0.2, niter=100, save_head=None, njob=-1):
    start = time.time()

    print('average evaluation')
    print('njob:', njob)

    def process(wrap, xs, ys, test_size, random):
        if ys is None:
            x_train, x_tests = train_test_split(xs, test_size=test_size, shuffle=True, random_state=100 + random)
            y_train = None
            y_tests = None
        else:
            x_train, x_tests, y_train, y_tests = train_test_split(xs, ys, test_size=test_size, shuffle=100 + random, random_state=None)
        estimator = clone(wrap)
        estimator.fit(x_train, y_train)
        eval = estimator.evaluation(x_tests, y_tests, verbose=0)
        gc.collect()
        return eval

    if njob != 1:
        # parallel
        evals = Parallel(n_jobs=njob)(delayed(process)(wrap, xs, ys, test_size, i) for i in tqdm(range(niter)))
    else:
        # serial (mainly to computes time)
        times = []
        evals = []
        for i in tqdm(range(niter)):
            s = time.time()
            eval = process(wrap, xs, ys, test_size, i)
            evals.append(eval)
            times.append(time.time() - s)
        print('running time:', time2str(float(np.average(times))))
        print('running time:', '$ %.3f \pm %.3f $' % (round(float(np.average(times)), 3), round(float(np.std(times)), 3)))
        print('running time:', '$ %f \pm %f $' % (float(np.average(times)), float(np.std(times))))

    keys = list(evals[0].keys())

    # scores for each keys
    resutls = concatinating_evals(evals)
    aves = []
    stds = []
    for key in keys:
        scores = resutls[key]

        ave = float(np.average(scores))
        std = float(np.std(scores))

        aves.append(ave)
        stds.append(std)

        # print for latex
        print()
        print('[', key, ']')
        print('$ %.3f \pm %.3f $' % (round(ave, 3), round(std, 3)))
        print('$ %f \pm %f $' % (ave, std))
        print(scores)

    # save log
    if save_head is not None:
        fpath = save_head + '_av_eval.csv'
        with open(fpath, 'w') as f:
            # head
            f.write('i,' + ','.join(keys) + '\n')
            f.write('ave,' + ','.join([str(ave) for ave in aves]) + '\n')
            f.write('std,' + ','.join([str(ave) for ave in stds]) + '\n')
            for i in range(niter):
                f.write('%3d,' % i + ','.join([str(evals[i][key]) for key in keys]) + '\n')
    print()
    print('all done:', time2str(time.time() - start))

    return resutls


def concatinating_evals(evals: [{str, float}]) -> {str, np.ndarray}:
    keys = list(evals[0].keys())
    resutls = {}
    for key in keys:
        scores = [e[key] for e in evals]
        scores = np.array(scores)
        resutls[key] = scores
    return resutls
