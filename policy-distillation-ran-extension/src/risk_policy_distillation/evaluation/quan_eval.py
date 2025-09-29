import logging
from fairlearn.metrics import demographic_parity_difference

import numpy as np
import pandas as pd
from click.formatting import iter_rows
from scipy.stats import stats
from transformers.trainer_pt_utils import metrics_format

logger = logging.getLogger('logger')
from tqdm import tqdm


def evaluate_performance(dataset, res_column, label_col, label_names, flip_labels=False):
    ground_truth = dataset[label_col]
    if flip_labels:
        ground_truth = 1 - ground_truth

    if isinstance(dataset[res_column][0], str):
        dataset[res_column] = dataset[res_column].apply(lambda x: label_names.index(x))

    return np.mean(ground_truth == dataset[res_column])


def evaluate_dataset(expl, model, dataset, input_column, expl_label, guard_label, path):
    logger.info('Evaluating on test dataset...')

    try:
        results_ds = pd.read_csv(path, header=0)
        if len(results_ds) == len(dataset):
            logger.info('Loaded dataset.')
            return results_ds

    except FileNotFoundError:
        results_ds = pd.DataFrame()

    results = []

    if 'index' not in dataset.columns:
        dataset = dataset.reset_index()

    for i, row in tqdm(dataset.iterrows()):
        if len(results_ds) and row['index'] in results_ds.index.values:
            logger.info('Loaded from dataset index = {}'.format(row.index))
            continue

        logger.info('Loaded {} prompts from dataset'.format(i))

        x = row[input_column]

        expl_answer = expl.predict(x)
        model_answer = model.ask_guardian(x)

        results.append((*row, expl_answer, model_answer))

        result_ds = pd.concat([results_ds, pd.DataFrame(results, columns=list(dataset.columns) + [expl_label, guard_label])])
        result_ds.to_csv(path, index=False)

    return result_ds

def perf_degradation(dataset, expl_label, guard_label, true_label, label_names):
    logger.info('Evaluating performance degradation...')
    model_correct = 0.0
    expl_correct = 0.0

    total = len(dataset)

    for i, row in tqdm(dataset.iterrows()):
        model_answer = label_names.index(row[guard_label])
        expl_answer = row[expl_label]
        true_y = row[true_label]

        model_correct += model_answer == true_y
        expl_correct += expl_answer == true_y

    perf_degr = model_correct / total - expl_correct / total

    return perf_degr


def fidelity(dataset, expl_label, guard_label, label_names):
    logger.info('Evaluating fidelity...')
    correct = 0.0
    total = len(dataset)

    if total == 0:
        return 0, 0

    tp, fp, fn = 0.0, 0.0, 0.0

    for i, row in tqdm(dataset.iterrows()):
        model_answer = label_names.index(row[guard_label])
        expl_answer = row[expl_label]

        correct += model_answer == expl_answer

        tp += model_answer == 1 and expl_answer == 1
        fp += model_answer == 0 and expl_answer == 1
        fn += model_answer == 1 and expl_answer == 0

    if tp == 0 and fp == 0 and fn == 0:
        return 0,0

    accuracy = correct / total
    f1 = 2 * tp / (2*tp + fp + fn)

    return accuracy, f1


def fidelity_array(dataset, expl_label, guard_label):
    fid_array = []

    for i, row in tqdm(dataset.iterrows()):
        model_answer = row[guard_label]
        expl_answer = row[expl_label]

        fid_array.append(int(model_answer == expl_answer))

    return fid_array


def fairness(dataset, category_label, expl_label, guard_label):
    logger.info('Evaluating fairness...')

    dataset[category_label].fillna(0, inplace=True)

    dataset[category_label] = dataset[category_label].astype('category').cat.codes

    expl_answers = dataset[expl_label].values
    guard_answers = dataset[guard_label].values
    categories = dataset[category_label]

    fairness = demographic_parity_difference(expl_answers,
                                             guard_answers,
                                             sensitive_features=categories)

    return fairness


def significant(m_1, m_2):
    t_statistic, p_value = stats.ttest_ind(m_1, m_2, equal_var=True)

    if p_value < 0.05:
        return True

    return False



