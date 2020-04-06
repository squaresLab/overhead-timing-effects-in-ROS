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


def get_fns_from_rows(cursor: sqlite3.Cursor, log_dir: str) -> List[Tuple[str, str]]:
    rows = cursor.fetchall()
    log_fns = []
    for row in rows:
        # logging.debug(row)
        log_fn = os.path.join(log_dir, f"{row[0]}.tlog")
        mutation_fn = row[7]
        # logging.debug(log_fn)
        log_fns.append((log_fn, mutation_fn))
    return log_fns


def get_from_db(log_db: str, log_dir: str = "../bags") -> Dict[str, List[Tuple[str,str]]]:
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


def convert_logs(log_fns: List[Tuple[str, str]],
                 field_type: str="GLOBAL_POSITION_INT") -> List[Tuple[str, str, np.array]]:
    one_field_lists = []
    fn_count = 0
    total_fns = len(log_fns)
    logging.debug(f"convert_logs log_fns: {log_fns}")
    for log_fn, mutation_fn in log_fns:
        logging.debug(f"convert_logs log_fn: {log_fn}")
        # convert the tlog to json, if it's not already json
        log_json = get_json(log_fn)

        one_field_dict = [x["data"] for x in log_json if
                          x["meta"]["type"] == field_type]
        timestamp_list = [x["meta"]["timestamp"] for x in log_json if
                          x["meta"]["type"] == field_type]
        assert(len(one_field_dict) == len(timestamp_list))

        if field_type == "GLOBAL_POSITION_INT":
            one_field_list = ([(x[0]['time_boot_ms'], x[0]['lat'], x[0]['lon'],
                                 x[0]['alt'], x[0]['relative_alt'], x[0]['vx'],
                                 x[0]['vy'], x[0]['vz'], x[0]['hdg'], x[1])
                                for x in zip(one_field_dict, timestamp_list)])

        elif field_type == "MISSION_ITEM_REACHED":
            one_field_list = ([(x[0]['seq'], x[1])
                               for x in zip(one_field_dict, timestamp_list)])
        elif field_type == "MISSION_CURRENT":
            one_field_list = ([(x[0]['seq'], x[1])
                               for x in zip(one_field_dict, timestamp_list)])
        else:
            logging.error(f"Field: {field_type} not supported")
            raise NotImplementedError
        one_field_np = np.array(one_field_list)
        one_field_lists.append((log_fn, mutation_fn, one_field_np))
        fn_count += 1
        logging.debug(f"File {fn_count} of {total_fns}")
    return one_field_lists


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


def logs_to_np(log_fns: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Tuple[str, str, Any]]]:
    all_logs = dict()
    # Turn the log files into np.arrays of the data
    for label, log_fns_subset in log_fns.items():
        logging.debug(f"label: {label}, log_fns_subset: {log_fns_subset}")
        logs_data = convert_logs(log_fns_subset)
        all_logs[label] = logs_data

    # logging.debug(f"all_logs: {all_logs}")
    return all_logs


def to_json_ready(all_logs: Dict[str, List[Tuple[str, str, np.array]]]) -> Dict[str, List[Tuple[str, str, List]]]:
    json_ready = dict()
    for key in all_logs:
        json_ready_list = []
        for item in all_logs[key]:
            json_ready_list.append((item[0], item[1], item[2].tolist()))
        json_ready[key] = json_ready_list
    return json_ready


def json_to_np(json_logs: Dict[str, List[Tuple[str, str, List]]]) -> Dict[str, List[Tuple[str, str, np.array]]]:
    all_logs = dict()
    for key in json_logs:
        all_logs_list = []
        for item in json_logs[key]:
            all_logs_list.append((item[0], item[1], np.array(item[2])))
        all_logs[key] = all_logs_list
    return all_logs


def memoize_log_db(log_db: str) -> Dict[str, List[Tuple[str, str, np.array]]]:
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


def get_logs(args: argparse.Namespace) -> Dict[str, List[Tuple[str, str, np.array]]]:
    log_fns: Dict[str, List[Tuple[str, str]]] = dict()

    logging.debug(f"args.log_fn: {args.log_fn}")

    # if args.log_fn and len(args.log_fn) > 1:
    if args.log_fn and len(args.log_fn) > 0:
        log_fns = insert(log_fns, "manual", [(x, "None") for x in args.log_fn] )
        logging.debug(f"log_fns: {log_fns}")
        all_logs = logs_to_np(log_fns)
    elif args.log_db:
        #TODO: This assumes all the logs in the database have the same
        # mission, which isn't always a good assumption. Fix it.
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
