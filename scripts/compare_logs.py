import argparse
import json
import os
import shlex
import sqlite3
import subprocess
import time
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple

from fastdtw import fastdtw as fdtw
import numpy as np
from scipy.spatial.distance import sqeuclidean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_fn", type=str, action="append")
    parser.add_argument("--use_nominal", action="store_true", default=False)
    parser.add_argument("--alg", type=str)
    parser.add_argument("--log_db", type=str)
    args = parser.parse_args()
    return args


def get_json(log_fn: str) -> List[Dict]:
    json_data = []
    if log_fn.endswith(".json"):
        json_fn = log_fn
    if log_fn.endswith(".tlog"):
        cmd = f"mavlogdump.py --format json {log_fn}"
        json_fn = f"{log_fn}.json"
        with open(json_fn, 'w') as json_file:
            print(cmd)
            subprocess.Popen(shlex.split(cmd), stdout=json_file)
    time.sleep(1)
    with open(json_fn, 'r') as json_read:
        print(json_fn)
        for line in json_read:
            json_line = json.loads(line)
            json_data.append(json_line)
    return json_data


def pairwise(ts_a: np.array, ts_b: np.array, weights: np.array,
             alg='eros') -> float:
    if alg == 'eros':
        return eros_similarity(ts_a, ts_b, weights)
    elif alg == 'fdtw':
        return fastdtw(ts_a, ts_b)
    else:
        raise NotImplementedError


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

    return np.sqrt(2 - 2 * eros)


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
    print("Value of w: %s" % w)
    assert np.isclose(1.0, np.sum(w)), 'w must sum to 1.0'
    assert all(w_i >= 0.0 for w_i in w)

    return w


def access_bag_db(db_fn: str) -> Tuple[sqlite3.Cursor, sqlite3.Connection]:
    conn = sqlite3.connect(db_fn)
    if conn is not None:
        c = conn.cursor()
    else:
        raise("Error! Cannot create database connection!")
    return c, conn

def get_fns_from_rows(cursor: sqlite3.Cursor, log_dir: str) -> List[str]:
    rows = cursor.fetchall()
    log_fns = []
    for row in rows:
        print(row)
        log_fn = os.path.join(log_dir, f"{row[0]}.tlog")
        print(log_fn)
        log_fns.append(log_fn)
    return log_fns


def get_from_db(log_db: str, log_dir: str = "../bags") -> Dict[str, List[str]]:
    log_fns: Dict[str, List[np.array]] = dict()
    cursor, conn = access_bag_db(log_db)

    filename="None"
    cursor.execute("select * from bagfns where mutation_fn=?", (filename,))
    nominal_fns = get_fns_from_rows(cursor, log_dir)
    log_fns['nominal'] = nominal_fns
    cursor.execute("select * from bagfns where mutation_fn!=?", (filename,))
    experimental_fns = get_fns_from_rows(cursor, log_dir)
    log_fns['experimental'] = experimental_fns


    conn.close()
    return log_fns


def convert_logs(log_fns: List[str]) -> List[np.array]:
    global_pos_lists = []
    for log_fn in log_fns:
        # convert the tlog to json, if it's not already json
        log_json = get_json(log_fn)
        #print(log_json)
        global_pos = [x["data"] for x in log_json if x["meta"]["type"] == "GLOBAL_POSITION_INT"]
        global_pos_list = ([(x['time_boot_ms'], x['lat'], x['lon'],
                             x['alt'], x['relative_alt'], x['vx'],
                             x['vy'], x['vz'], x['hdg']) for x in
                            global_pos])
        #print(global_pos_list)
        global_np = np.array(global_pos_list)
        global_pos_lists.append(global_np)

    return global_pos_lists


def get_logs(args: argparse.Namespace)-> Tuple[List[np.array], List[str]]:
    log_fns: List[np.array] = list()
    labels = list()

    if args.log_fn and len(args.log_fn) > 1:
        log_fns.extend(args.log_fn)
        labels.extend(['manual'] * len(args.log_fn))
    elif args.log_db:
        logs = get_from_db(args.log_db, args.log_dir)
        for label in logs:
            log_fns.extend(logs[label])
            labels.extend([label] * len(logs[label]))

    # Turn the log files into np.arrays of the data
    # TODO
    logs_data = convert_logs(log_fns)

    return logs_data, labels


def get_comparisons(logs: List[np.array],
                    weights: np.array = None, alg: str = 'eros') -> np.array:
    comparisons = np.zeros(len(logs), len(logs))
    for i in range(len(logs)):
        for j in range(len(logs)):
            comparison = pairwise(logs[i], logs[j], weights=weights,
                                  alg=alg)
        comparisons[i][j] = comparison
    return comparisons


def compare_logs(logs: List[np.array], labels: List[str],
                 alg: str = 'eros') -> Dict[str, np.array]:
    assert(np.size(0,) == len(labels))
    label_logs = zip(labels, logs)

    # Get all the groups for individual comparisons, by labels
    label_set = set(labels)
    logs_dict = dict()
    for label in labels:
        logs_subset = [x[1] for x in label_logs if x[0] == label]
        logs_dict[label] = logs_subset

    if alg == 'eros':
        n = len(logs[0])
        assert(all([len(x) == n for x in logs]))
        weights = eros_weights(logs)
    else:
        weights = None

    all_comparisons = dict()
    for label, logs_subset in (list(logs_dict.items()) +
                               list([('total', logs)])):
        if len(logs_subset) > 1:
            comparisons = get_comparisons(logs, weights, alg)
            all_comparisons[label] = comparisons

    return all_comparisons

def main():
    args = parse_args()

    # get the logs to compare
    logs, labels = get_logs(args)

    # compare the logs
    comparisons = compare_logs(logs, labels)

    """
    global_pos_lists = []

    if args.log_fn and len(args.log_fn) > 0:
        log_fns = args.log_fn
    elif args.use_nominal:
        nominal_log_fns = get_nominal(args.log_db)
        log_fns = nominal_log_fns

    for log_fn in log_fns:
        # convert the tlog to json, if it's not already json
        log_json = get_json(log_fn)
        #print(log_json)
        global_pos = [x["data"] for x in log_json if x["meta"]["type"] == "GLOBAL_POSITION_INT"]
        global_pos_list = ([(x['time_boot_ms'], x['lat'], x['lon'],
                             x['alt'], x['relative_alt'], x['vx'],
                             x['vy'], x['vz'], x['hdg']) for x in
                            global_pos])
        #print(global_pos_list)
        global_np = np.array(global_pos_list)
        global_pos_lists.append(global_np)

    comparisons = np.zeros((len(global_pos_lists), len(global_pos_lists)))
    print(comparisons.shape)
    weights = eros_weights(global_pos_lists)
    if len(comparisons) > 1:
        for i in range(len(global_pos_lists)):
            for j in range(len(global_pos_lists)):
                comparison = pairwise(global_pos_lists[i],
                                      global_pos_lists[j], weights,
                                      alg=args.alg)
                #print(comparison)
                comparisons[i][j] = comparison

    print(comparisons)
"""


if __name__ == '__main__':
    main()
