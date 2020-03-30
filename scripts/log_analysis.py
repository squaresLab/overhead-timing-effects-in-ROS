import argparse
import datetime
import json
import logging
import os
import random
import shlex
import sqlite3
import subprocess
import time
from typing import Any, Dict, List, Tuple

import numpy as np


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
        # logging.debug(row)
        log_fn = os.path.join(log_dir, f"{row[0]}.tlog")
        # logging.debug(log_fn)
        log_fns.append(log_fn)
    return log_fns


def get_from_db(log_db: str, log_dir: str = "../bags") -> Dict[str, List[str]]:
    log_fns: Dict[str, List[np.array]] = dict()
    cursor, conn = access_bag_db(log_db)

    # TODO - DEBUG HERE -- ARE WE GETTING ALL THE FILES?
    # Keep track of the variations in the varied ones

    filename = "None"
    cursor.execute("select * from bagfns where mutation_fn=?", (filename,))
    nominal_fns = get_fns_from_rows(cursor, log_dir)
    log_fns['nominal'] = nominal_fns
    logging.debug(f"len(nominal_fns): {len(nominal_fns)}")
    logging.debug(f"len(log_fns['nominal']): {len(log_fns['nominal'])}")

    cursor.execute("select * from bagfns where mutation_fn!=?", (filename,))
    experimental_fns = get_fns_from_rows(cursor, log_dir)
    log_fns['experimental'] = experimental_fns

    conn.close()
    return log_fns


def get_json(log_fn: str) -> List[Dict]:
    logging.debug(f"log_fn: {log_fn}")
    json_data = []
    if log_fn.endswith(".json"):
        json_fn = log_fn
    elif log_fn.endswith(".tlog"):
        json_fn = f"{log_fn}.json"
        if not os.path.isfile(json_fn):
            cmd = f"mavlogdump.py --format json {log_fn}"
            with open(json_fn, 'w') as json_file:
                logging.debug(cmd)
                subprocess.Popen(shlex.split(cmd), stdout=json_file)
            time.sleep(1)
    else:
        logging.debug(f"What does this end with?: {log_fn}")
        raise NotImplementedError
    with open(json_fn, 'r') as json_read:
        logging.debug(json_fn)
        for line in json_read:
            json_line = json.loads(line)
            json_data.append(json_line)
    return json_data


def convert_logs(log_fns: List[str]) -> List[Tuple[str, np.array]]:
    global_pos_lists = []
    fn_count = 0
    total_fns = len(log_fns)
    logging.debug(f"convert_logs log_fns: {log_fns}")
    for log_fn in log_fns:
        logging.debug(f"convert_logs log_fn: {log_fn}")
        # convert the tlog to json, if it's not already json
        log_json = get_json(log_fn)
        global_pos = [x["data"] for x in log_json if
                      x["meta"]["type"] == "GLOBAL_POSITION_INT"]
        global_pos_list = ([(x['time_boot_ms'], x['lat'], x['lon'],
                             x['alt'], x['relative_alt'], x['vx'],
                             x['vy'], x['vz'], x['hdg']) for x in
                            global_pos])
        global_np = np.array(global_pos_list)
        global_pos_lists.append((log_fn, global_np))
        fn_count += 1
        logging.debug(f"File {fn_count} of {total_fns}")
    return global_pos_lists


def equalize_lengths(logs: List[np.array], length: int = 0) -> List[np.array]:
    if length == 0:
        length = min([len(x) for x in logs])

    new_logs = []
    for log in logs:
        while len(log) > length:
            to_delete = random.randrange(len(log))
            log = np.delete(log, to_delete, axis=0)
        new_logs.append(log)

    # logging.debug(str([x for x in new_logs]))
    return new_logs


def logs_to_np(log_fns: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, Any]]]:
    all_logs = dict()
    # Turn the log files into np.arrays of the data
    for label, log_fns_subset in log_fns.items():
        logging.debug(f"label: {label}, log_fns_subset: {log_fns_subset}")
        logs_data = convert_logs(log_fns_subset)
        all_logs[label] = logs_data

    # logging.debug(f"all_logs: {all_logs}")
    return all_logs


def to_json_ready(all_logs: Dict[str, List[Tuple[str, np.array]]]) -> Dict[str, List[Tuple[str, List]]]:
    json_ready = dict()
    for key in all_logs:
        json_ready_list = []
        for item in all_logs[key]:
            json_ready_list.append((item[0], item[1].tolist()))
        json_ready[key] = json_ready_list
    return json_ready


def json_to_np(json_logs: Dict[str, List[Tuple[str, List]]]) -> Dict[str, List[Tuple[str, np.array]]]:
    all_logs = dict()
    for key in json_logs:
        all_logs_list = []
        for item in json_logs[key]:
            all_logs_list.append((item[0], np.array(item[1])))
        all_logs[key] = all_logs_list
    return all_logs


def memoize_log_db(log_db: str) -> Dict[str, List[Tuple[str, np.array]]]:
    date = time.strftime("%m-%d-%Y")
    memoized_fn = f"{log_db}_{date}.json"
    logging.debug(memoized_fn)

    if not os.path.isfile(memoized_fn):
        log_fns = get_from_db(log_db)
        all_logs = logs_to_np(log_fns)
        with open(memoized_fn, 'w') as memoized_file:
            json_ready_logs = to_json_ready(all_logs)
            json.dump(json_ready_logs, memoized_file)
    else:
        with open(memoized_fn, 'r') as memoized_file:
            json_logs = json.load(memoized_file)
            all_logs = json_to_np(json_logs)
    return all_logs


def get_logs(args: argparse.Namespace) -> Dict[str, List[Tuple[str, np.array]]]:
    log_fns: Dict[str, List[str]] = dict()

    logging.debug(f"args.log_fn: {args.log_fn}")

    # if args.log_fn and len(args.log_fn) > 1:
    if args.log_fn and len(args.log_fn) > 0:
        log_fns = insert(log_fns, "manual", args.log_fn)
        logging.debug(f"log_fns: {log_fns}")
        all_logs = logs_to_np(log_fns)
    elif args.log_db:
        all_logs = memoize_log_db(args.log_db)

    return all_logs


def insert(db: Dict[str, List[Any]], label: str,
           data: List[Any]) -> Dict[str, List[Any]]:
    logging.debug(db)
    if label in db:
        db[label] = db[label] + data
    else:
        db[label] = data
    return db
