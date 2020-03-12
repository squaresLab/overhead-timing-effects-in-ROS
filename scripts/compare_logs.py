import argparse
import json
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
#np.set_printoptions(threshold=sys.maxsize)
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
        json_fn = f"{log_fn}.json"
        if not os.path.isfile(json_fn):
            cmd = f"mavlogdump.py --format json {log_fn}"
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
        comp = eros_similarity(ts_a, ts_b, weights)
    elif alg == 'fdtw':
        comp = fastdtw(ts_a, ts_b)
    else:
        raise NotImplementedError
    print(f"ts_a: {ts_a}")
    print(f"ts_b: {ts_b}")
    print(f"alg: {alg} comp: {comp}")
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
        cursor = conn.cursor()
    else:
        raise("Error! Cannot create database connection!")
    return cursor, conn


def get_fns_from_rows(cursor: sqlite3.Cursor, log_dir: str) -> List[str]:
    rows = cursor.fetchall()
    log_fns = []
    for row in rows:
        # print(row)
        log_fn = os.path.join(log_dir, f"{row[0]}.tlog")
        # print(log_fn)
        log_fns.append(log_fn)
    return log_fns


def memoize_dst(log_fn_1, log_fn_2, db_fn):
    # Access the database
    cursor, conn = access_bag_db(db_fn)

    # Check if the database has the correct table.
    # TODO
    # If not, create the correct table.
    # TODO
    # Check if the distance is in the table
    # TODO
    # If so return it
    # TODO
    # If not, calculate it

def get_from_db(log_db: str, log_dir: str = "../bags") -> Dict[str, List[str]]:
    log_fns: Dict[str, List[np.array]] = dict()
    cursor, conn = access_bag_db(log_db)

    filename="None"
    cursor.execute("select * from bagfns where mutation_fn=?", (filename,))
    nominal_fns = get_fns_from_rows(cursor, log_dir)
    log_fns['nominal'] = nominal_fns
    print(f"len(nominal_fns): {len(nominal_fns)}")
    print(f"len(log_fns['nominal']): {len(log_fns['nominal'])}")

    cursor.execute("select * from bagfns where mutation_fn!=?", (filename,))
    experimental_fns = get_fns_from_rows(cursor, log_dir)
    log_fns['experimental'] = experimental_fns

    conn.close()
    return log_fns


def convert_logs(log_fns: List[str]) -> List[np.array]:
    global_pos_lists = []
    fn_count = 0
    total_fns = len(log_fns)
    for log_fn in log_fns:
        # convert the tlog to json, if it's not already json
        log_json = get_json(log_fn)
        global_pos = [x["data"] for x in log_json if x["meta"]["type"] == "GLOBAL_POSITION_INT"]
        global_pos_list = ([(x['time_boot_ms'], x['lat'], x['lon'],
                             x['alt'], x['relative_alt'], x['vx'],
                             x['vy'], x['vz'], x['hdg']) for x in
                            global_pos])
        global_np = np.array(global_pos_list)
        global_pos_lists.append(global_np)
        fn_count += 1
        print(f"File {fn_count} of {total_fns}")
    return global_pos_lists


def insert(db: Dict[str, List[Any]], label: str, data: List[Any]) -> Dict[str, List[Any]]:
    print(db)
    if label in db:
        db[label] = db[label] + data
    else:
        db[label] = data
    return db


def get_logs(args: argparse.Namespace)-> Dict[str, List[np.array]]:
    log_fns: Dict[str, List[np.array]] = dict()

    if args.log_fn and len(args.log_fn) > 1:
        log_fns = insert(log_fns, "manual", args.log_fn)
    elif args.log_db:
        log_fns = get_from_db(args.log_db)

    all_logs = dict()
    # Turn the log files into np.arrays of the data
    for label, log_fns_subset in log_fns.items():
        logs_data = convert_logs(log_fns_subset)
        all_logs[label] = logs_data

    print(f"all_logs: {all_logs}")
    return all_logs


def get_comparisons(logs: List[np.array],
                    weights: np.array = None, alg: str = 'eros') -> np.array:
    comparisons = np.zeros((len(logs), len(logs)))
    for i in range(len(logs)):
        for j in range(len(logs)):
            if comparisons[i][j] == 0:
                print(f"comparing log[{i}] and log[{j}]")
                comparison = pairwise(logs[i], logs[j], weights=weights,
                                      alg=alg)
                comparisons[i][j] = comparison
                comparisons[j][i] = comparison
    return comparisons


def equalize_lengths(logs: List[np.array], length: int=0) -> List[np.array]:
    if length == 0:
        length = min([len(x) for x in logs])

    new_logs = []
    for log in logs:
        while len(log) > length:
            to_delete = random.randrange(len(log))
            log = np.delete(log, to_delete, axis=0)
        new_logs.append(log)

    #print(str([x for x in new_logs]))
    return new_logs

def compare_logs(logs_dict: Dict[str, List[np.array]],
                 alg: str = 'eros') -> Dict[str, np.array]:
    #assert(len(logs) == len(labels)), f"{len(logs)} {len(labels)}"
    #label_logs = zip(labels, logs)

    # Get all the groups for individual comparisons, by labels
    #label_set = set(labels)
    #logs_dict = dict()
    #for label in label_set:
    #    logs_subset = [x[1] for x in label_logs if x[0] == label]
        # print(f"label: {label} logs_subset: {logs_subset}")
    #    logs_dict[label] = logs_subset
    #    print(f"len(logs_dict[{label}]): {len(logs_dict[label])}")

    print(f"logs_dict.keys(): {logs_dict.keys()}")

    all_logs = []
    for label, logs in logs_dict.items():
        all_logs.extend(logs)

    if alg == 'eros':
        n = len(all_logs[0])
        # print(logs[0])
        if not all([len(x) == n for x in all_logs]):
            logs_eq = equalize_lengths(all_logs)
        else:
            logs_eq = all_logs
        weights = eros_weights(logs_eq)
    else:
        weights = None

    print(f"logs_dict.keys(): {logs_dict.keys()}")
    all_comparisons = dict()
    for label, logs_subset in (list(logs_dict.items())): # +
                               #list([('total', logs)])):
        print(f"label: {label} len(logs_subset): {len(logs_subset)}")
        if len(logs_subset) > 1:
            comparisons = get_comparisons(logs, weights, alg)
            all_comparisons[label] = comparisons

    return all_comparisons


def main():
    args = parse_args()

    # get the logs to compare
    logs_dict = get_logs(args)

    # compare the logs
    comparisons = compare_logs(logs_dict, args.alg)
    print(f"comparisons: {comparisons}")


if __name__ == '__main__':
    main()
