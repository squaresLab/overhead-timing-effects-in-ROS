import argparse
import json
import logging
import os
import random
import shlex
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple

from fastdtw import fastdtw as fdtw
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
from scipy.spatial.distance import sqeuclidean

import log_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_fn", type=str, action="append")
    parser.add_argument("--use_nominal", action="store_true", default=False)
    parser.add_argument("--alg", type=str)
    parser.add_argument("--log_db", type=str)
    parser.add_argument("--out_fn", type=str, default="comparisons.out")
    parser.add_argument("--logging", type=str, default="compare_log.out")
    args = parser.parse_args()
    return args


def identical(ts_a: np.array, ts_b: np.array) -> bool:
    return np.array_equal(ts_a, ts_b)


def pairwise(ts_a: np.array, ts_b: np.array, weights: np.array,
             alg='eros') -> float:
    if identical(ts_a, ts_b):
        logging.warn(f"ts_a and ts_b are identical!!")
        # logging.debug(f"ts_a: {ts_a}")
        # logging.debug(f"ts_b: {ts_b}")

    if alg == 'eros':
        comp = eros_distance(ts_a, ts_b, weights)
    elif alg == 'fdtw' or 'fastdtw':
        comp = fastdtw(ts_a, ts_b)
    else:
        raise NotImplementedError
    logging.debug(f"alg: {alg} comp: {comp}")
    return comp


def fastdtw(x: np.array, y: np.array) -> float:
    return np.sqrt(fdtw(x, y, dist=sqeuclidean)[0])


# Borrowed from Afsoon
def eros_similarity(ts_a: np.array, ts_b: np.array, w: np.array) -> float:
    """Computes the Eros similarity between two multivariate time series."""
    # TODO how do we deal with floating point errors?
    assert np.isclose(1.0, np.sum(w)), 'w must sum to 1.0'

    # ensure that columns represent features and rows represent observations
    n = len(w)
    assert np.size(ts_a, 1) == n
    assert np.size(ts_b, 1) == n

    # compute covariance matrices
    m_a = np.cov(ts_a, rowvar=False)
    m_b = np.cov(ts_b, rowvar=False)

    # compute right eigenvector matrices
    v_a = np.linalg.svd(m_a)[0]
    v_b = np.linalg.svd(m_b)[0]

    sum_ = 0.0
    for i, w_i in enumerate(w):
        sum_ += w_i * np.abs(np.inner(v_a[i], v_b[i]))
        # logging.debug(f"eros_similarity sum_: {sum_}")

    # avoid rounding errors
    if np.isclose(sum_, 1.0):
        sum_ = 1.0
    elif np.isclose(sum_, 0.0):
        sum_ = 0.0

    assert 0.0 <= sum_ <= 1.0
    return sum_


def eros_distance(ts_a: np.array, ts_b: np.array, w: np.array) -> float:
    """Computes the Eros distance between two multivariate time series."""
    eros = eros_similarity(ts_a, ts_b, w)
    logging.debug(f"eros_similarity: {eros}")
    eros_distance = np.sqrt(2 - 2 * eros)
    logging.debug(f"eros_distance: {eros_distance}")
    return eros_distance


def eros_weights(dataset: Collection[np.array],
                 *,
                 normalize: bool = True,
                 agg: Callable[[Sequence[float]], float] = np.sum
                 ) -> np.array:
    """
    Computes a weight vector w for a given set of time series based on the
    distribution of eigenvalues.
    Parameters
    ----------
    dataset: Collection[np.array]
        A collection of M multivariate time series, each containing N
        variables.
    normalize: bool
        If :code:`True`, eigenvalues will be normalized; if :code:`False`, raw
        eigenvalues will be used.
    agg: Callable[[Sequence[float]], float]
        The aggregation function that should be used to transform a sequence of
        eigenvalues into a scalar value.
    Returns
    -------
    np.array
        A weight vector of length N.
    """
    # determine number of time series
    m = len(dataset)
    assert m > 0, "at least one time series must be provided."

    # determine number of variables (assuming columns represent variables)
    # check that each time series contains the same number of variables
    n = np.size(list(dataset)[0], 1)
    assert all(np.size(ts, 1) == n for ts in dataset)

    # build an N*M matrix where each column stores the eigenvalues for
    # a separate time series
    s = np.empty((n, m))
    for i, ts in enumerate(dataset):
        s[:, i] = np.linalg.eig(np.cov(ts, rowvar=False))[0]

    # if enabled, normalize the eigenvalues in each column
    if normalize:
        for i in range(m):
            summed = np.sum(s[:, i])
            if summed != 0.0:
                s[:, i] /= summed

    # build the weight vector
    w = np.empty(n)
    for i in range(n):
        w[i] = agg(s[i, :])
    if np.any(w):
        w /= np.sum(w)
    else:
        # All ws are zero, then they all have same weight
        w = np.full(n, 1.0/n)
    w[np.isclose(w, 0.0)] = 0.0

    # check that assumptions hold
    logging.debug("Value of w: %s" % w)
    assert np.isclose(1.0, np.sum(w)), 'w must sum to 1.0'
    assert all(w_i >= 0.0 for w_i in w)

    return w


def memoize_dst(log_1: Tuple[str, str], log_2: Tuple[str, str], dist_fn: str,
                db_fn: str, weights: np.array = None) -> float:

    log_fn_1, log_data_1 = log_1
    log_fn_2, log_data_2 = log_2
    # Access the database
    cursor, conn = log_analysis.access_bag_db(db_fn)

    # Check if the database has the correct table.
    # If not, create the correct table.
    sql_create_dist_table = """CREATE TABLE IF NOT EXISTS dist (
           log_fn_1 text,
           log_fn_2 text,
           dist_fn text,
           dist_value float
       ); """

    try:
        cursor.execute(sql_create_dist_table)
    except sqlite3.Error as e:
        logging.warning(e)

    # Check if the distance is in the table
    command = "SELECT * FROM dist WHERE log_fn_1=? AND log_fn_2=? AND dist_fn=?"
    cursor.execute(command, (log_fn_1, log_fn_2, dist_fn))
    rows = cursor.fetchall()
    logging.debug(f"rows: {rows}")
    if len(rows) == 1:
        logging.debug(f"distance: {rows[0]}")
        # If so return it
        print(f"rows: {rows}")
        print(f"type(rows[0][-1]): {type(rows[0][-1])}")
        dist_value = float(rows[0][-1])
    elif len(rows) == 0:
        # If not, calculate it
        dist_value = pairwise(log_data_1, log_data_2, weights, alg=dist_fn)

        logging.debug(f"values to insert: {log_fn_1}, {log_fn_2}, {dist_fn}, {dist_value}")
        # Insert it into the table
        command = "INSERT INTO dist VALUES (?, ?, ?, ?)"
        values = (log_fn_1, log_fn_2, dist_fn, dist_value)
        cursor.execute(command, values)
        # Assume reciprocal distances
        values = (log_fn_2, log_fn_1, dist_fn, dist_value)
        cursor.execute(command, values)
    else:
        logging.error(f"Too many rows returned: {rows}")
        raise NotImplementedError

    conn.commit()
    return dist_value


def get_comparisons(logs: List[np.array], db_fn: str,
                    weights: np.array = None, alg: str = 'eros') -> np.array:
    comparisons = np.zeros((len(logs), len(logs)))
    for i in range(len(logs)):
        for j in range(len(logs)):
            if comparisons[i][j] == 0 and i != j:
                logging.debug(f"comparing log[{i}] and log[{j}]")
                comparison = memoize_dst(logs[i], logs[j], alg, db_fn,
                                         weights=weights)
                comparisons[i][j] = comparison
                comparisons[j][i] = comparison
    return comparisons


def compare_logs(logs_dict: Dict[str, List[np.array]], db_fn: str,
                 alg: str = 'eros') -> Dict[str, np.array]:
    # assert(len(logs) == len(labels)), f"{len(logs)} {len(labels)}"
    # label_logs = zip(labels, logs)

    # Get all the groups for individual comparisons, by labels
    # label_set = set(labels)
    # logs_dict = dict()
    # for label in label_set:
    #    logs_subset = [x[1] for x in label_logs if x[0] == label]
        # logging.debug(f"label: {label} logs_subset: {logs_subset}")
    #    logs_dict[label] = logs_subset
    #    logging.debug(f"len(logs_dict[{label}]): {len(logs_dict[label])}")

    logging.debug(f"logs_dict.keys(): {logs_dict.keys()}")

    all_logs = []
    for label, logs in logs_dict.items():
        all_logs.extend(logs)

    if alg == 'eros':
        n = len(all_logs[0])
        # logging.debug(logs[0])
        if not all([len(x) == n for x in all_logs]):
            logs_eq = log_analysis.equalize_lengths(all_logs)
        else:
            logs_eq = all_logs
        # weights = eros_weights(logs_eq)
        weights = eros_weights(logs_dict["nominal"])
    else:
        weights = None

    logging.debug(f"logs_dict.keys(): {logs_dict.keys()}")
    all_comparisons = dict()
    for label, logs_subset in (list(logs_dict.items())):
        logging.debug(f"label: {label} len(logs_subset): {len(logs_subset)}")
        if len(logs_subset) > 1:
            comparisons = get_comparisons(logs_subset, db_fn,
                                          weights=weights, alg=alg)
            all_comparisons[label] = comparisons

    return all_comparisons


def main():
    args = parse_args()

    log_stream = logging.StreamHandler()
    log_file = logging.FileHandler(args.logging)
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(handlers=[log_stream, log_file], level=logging.DEBUG,
                        format=format_str, datefmt=date_str)

    # get the logs to compare
    logs_dict = log_analysis.get_logs(args)

    # compare the logs
    comparisons = compare_logs(logs_dict, args.log_db, args.alg)
    logging.debug(f"comparisons: {comparisons}")
    out_fn = args.out_fn
    for_json = dict()
    for key in comparisons:
        for_json[key] = comparisons[key].tolist()
    with open(out_fn, 'w') as outfile:
        json.dump(for_json, outfile)
    non_zero = np.nonzero(comparisons)
    logging.debug(f"non_zero: {non_zero}")


if __name__ == '__main__':
    main()
