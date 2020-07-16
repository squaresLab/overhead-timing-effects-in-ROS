import argparse
import datetime
import json
import logging
import math
import os
import random
import shlex
import sqlite3
import subprocess
import time
from typing import Any, Dict, List, Tuple
import yaml

import numpy as np
import rosbag


def euclidean_distance(a: Tuple[float, float, float],
                       b: Tuple[float, float, float],
                       log_type: str = "ardu") -> float:

    #print(type(a))
    #print(f"point a: {a}, point b: {b}")

    try:
        if log_type == "ardu":
            dist = math.sqrt((float(a[0]) - float(b[0]))**2 +
                             (float(a[1]) - float(b[1]))**2 +
                             (float(a[2]) - float(b[2]))**2)
        elif log_type == "husky":
            dist = math.sqrt((float(a[0]) - float(b[0]))**2 +
                             (float(a[1]) - float(b[1]))**2)

    except:
        logging.error(f"point a: {a}, point b: {b}")
        for i in a:
            logging.error(f"type: {type(i)}")
        for j in b:
            logging.error(f"type: {type(j)}")

    return dist


def reaches_waypoints(dist_dict: Dict[int, float],
                      mission: List[Tuple[float, float, float]],
                      log_type: str = "ardu",
                      tolerance: float = 0.05) -> Dict[int, bool]:
    """
    Does the robot hit all the waypoints (and return home for ardu)?
    Returns a dict of waypoint number to boolean, as to whether reached.
    (We'll need a measure of good enough --tolerance paramater?)
    """

    hits_all = dict([(x[0], x[1] <= tolerance) for x in dist_dict.items()])
    return hits_all


def completion_time(one_log: np.array,
                    log_type: str="ardu") -> float:
    assert(type(one_log))
    last_position = one_log[-1]
    return last_position[0]


def distance_to_each_waypoint(one_log: np.array,
                              mission: List[Tuple[float, float, float]],
                              log_type: str = "ardu",
                              final_dist: bool = True) -> Dict[int, float]:
    """
    What's the closest distance the robot gets to each waypoint?
    Returns a dict of waypoint number to distance (float).
    Note -- this doesn't account for going to the waypoints in order.
    It's just the closest at any point in the execution.
    If final_dist is true, replace the last waypoint distance
    with the end_distance_from_waypoint.

    """
    waypoint_dict = dict()
    for waypoint, index in zip(mission, range(len(mission))):
        #print(f"waypoint: {waypoint} index: {index}, one_log: {one_log}")
        dist = waypoint_to_log_dist(one_log, waypoint, log_type=log_type)
        waypoint_dict[index] = dist

    assert(len(mission) == len(waypoint_dict))
    #logging.debug(f"waypoint_dict before: {waypoint_dict}")

    if final_dist:
        max_waypoint_num = max(waypoint_dict.keys())
        last_dist = end_dist_from_waypoint(one_log, mission[max_waypoint_num],
                                           log_type=log_type)
        #logging.debug(f"last_dist: {last_dist}")
        waypoint_dict[max_waypoint_num] = last_dist

        # pick a distance to the first waypoint from the first half
        # of the log -- this is a sloppy approximation
        half_log_length = int(len(one_log)/2)
        half_log = one_log[:half_log_length]
        min_waypoint_num = min(waypoint_dict.keys())
        first_dist = waypoint_to_log_dist(half_log, mission[min_waypoint_num],
                                          log_type=log_type)
        #logging.debug(f"first_dist: {first_dist}")
        waypoint_dict[min_waypoint_num] = first_dist
    #logging.debug(f"waypoint_dict after: {waypoint_dict}")

    assert(len(mission) == len(waypoint_dict))

    return waypoint_dict


def end_dist_from_waypoint(one_log: np.array,
                           waypoint: Tuple[float, float, float],
                           log_type: str = "ardu") -> float:
    """
    How far is the last location in the log from the final waypoint?
    """
    last_location = one_log[-1]
    dist = euclidean_distance(
        (last_location[1], last_location[2], last_location[3]),
        waypoint,
        log_type=log_type)
    #logging.debug(f"end_dist_from_waypoint: {dist}")

    return dist


def closest_dist_to_waypoint_by_num(one_log: np.array,
                                    mission: List[Tuple[float, float, float]],
                                    waypoint_num: int,
                                    log_type: str = "ardu"):
    """
    What's the closest distance that the robot's path gets to a given
    waypoint (at any point in the run or at the appropriate point in the run)?
    """
    assert(waypoint_num < len(mission))
    waypoint = mission[waypoint_num]
    return waypoint_to_log_dist(waypoint, one_log, log_type)


def waypoint_to_log_dist(log: np.array,
                         waypoint: Tuple[float, float, float],
                         log_type: str = "ardu") -> float:
    """
    What's the closest distance that the robot's path gets to a given
    waypoint (at any point in the run or at the appropriate point in the run)?
    """
    #print(f"type(log): {type(log)}")
    min_dist = min([euclidean_distance((x[1], x[2], x[3]), waypoint,
                                       log_type=log_type)
                    for x in log])
    return min_dist


def end_distance(log: np.array, mission: List[Tuple[float, float, float]],
                 log_type: str = "ardu"):
    """
    The distance between the end point and the intended end point.

    """
    last_location = log[-1]
    last_waypoint = log[-1]
    return euclidean_distance(last_location, last_waypoint)


def nominal_vs_one_log(nominal_log, experimental_log, log_type: str = "ardu"):
    """
    Compare an experimental log against a representative nominal log
    (instead of a waypoint) with various metrics
    """
    raise NotImplementedError


def access_bag_db(db_fn: str) -> Tuple[sqlite3.Cursor, sqlite3.Connection]:
    conn = sqlite3.connect(db_fn)
    if conn is not None:
        cursor = conn.cursor()
    else:
        raise("Error! Cannot create database connection!")
    return cursor, conn


def get_fns_from_rows(cursor: sqlite3.Cursor, log_dir: str,
                      log_type: str = "ardu") -> List[Tuple[str, str, str]]:
    rows = cursor.fetchall()
    log_fns = []
    for row in rows:
        # logging.debug(row)
        if log_type == "ardu":
            log_fn = os.path.join(log_dir, f"{row[0]}.tlog")
        elif log_type == "husky":
            log_fn = os.path.join(log_dir, row[0])
            print(f"log_fn: {log_fn}")
        mutation_fn = row[7]
        mission_fn = row[5]
        # logging.debug(log_fn)
        log_fns.append((log_fn, mutation_fn, mission_fn))
    return log_fns


def get_from_db(log_db: str, log_dir: str = "../bags",
                log_type: str = "ardu") -> Dict[str, List[Tuple[str, str, str]]]:
    log_fns: Dict[str, List[np.array]] = dict()
    cursor, conn = access_bag_db(log_db)

    # TODO - Get separately for each mission
    # TODO - Find all the mission names then get them all separately

    # TODO - DEBUG HERE -- ARE WE GETTING ALL THE FILES?
    # Keep track of the variations in the varied ones

    cursor.execute("select * from bagfns")
    rows = cursor.fetchall()
    conn.close()
    for row in rows:
        log_fn, docker_image_sha, docker_image, container_uuid, mission_sha, mission_fn, mutation_sha, mutation_fn, sources = row
        #logging.info(f"mutation_fn: {mutation_fn}")
        if mutation_fn == "None":
            label = "nominal"
        else:
            label = "experimental"
        if log_type == "ardu":
            log_fn = os.path.join(log_dir, f"{log_fn}.tlog")
        elif log_type == "husky":
            log_fn = os.path.join(log_dir, log_fn)
        if label not in log_fns:
            log_fns[label] = []
        log_fns[label] = log_fns[label] + [(log_fn, mutation_fn, mission_fn)]


    # filename = "None"
    # cursor.execute("select * from bagfns where mutation_fn=?", (filename,))
    # nominal_fns = get_fns_from_rows(cursor, log_dir, log_type=log_type)
    # log_fns['nominal'] = nominal_fns
    #logging.debug(f"len(nominal_fns): {len(nominal_fns)}")
    logging.debug(f"len(log_fns['nominal']): {len(log_fns['nominal'])}")
    logging.debug(f"len(log_fns['experimental']): {len(log_fns['experimental'])}")
    #cursor.execute("select * from bagfns where mutation_fn!=?", (filename,))
    #experimental_fns = get_fns_from_rows(cursor, log_dir)
    #log_fns['experimental'] = experimental_fns


    return log_fns


def fn_topic_to_topic(fn_topic: str) -> str:
    topic = fn_topic.replace("-", "/")
    topic = topic.strip("_")
    return topic


def topic_to_fn_topic(topic: str) -> str:
    fn_topic = topic.replace("/", "-")
    return fn_topic


def mut_fn_to_topic_delay(fn: str, log_type: str = "ardu") -> Tuple[str, float]:
    if log_type == "husky":
        basename = fn.split("/")[-1]
        base_var = basename.replace("husky_waypoints_ground_truth_remap_", "")
        delay_split = base_var.split("_")
        try:
            delay_str = delay_split[-2]
            delay = float(delay_str)
        except IndexError:
            #logging.debug(f"cannot split filename.")
            delay = 0.0
            topic = "None"
            return topic, delay
        except ValueError:
            logging.debug(f"{delay_split[-2]} is not a float for a delay amount")
            delay = 0.0
            topic = "None"
            return topic, delay
        try:
            topic = base_var.partition(delay_str)[0]
            topic = fn_topic_to_topic(topic)
        except IndexError:
            topic = "None"

    elif log_type == "ardu":
        raise NotImplementedError

    return topic, delay


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
            try:
                json_line = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                logging.warn(f"Cannot read json in line: {line}")
                continue
            json_data.append(json_line)
    return json_data


def change_bag_base(log_fn: str, bag_base: str):
    bare_fn = os.path.basename(log_fn)
    new_fn = os.path.join(bag_base, bare_fn)
    return new_fn


def get_memoized_husky_data(log_fn: str, field_type: str) -> List[Any]:
    field_str = topic_to_fn_topic(field_type)
    memoized_fn = f"{log_fn}_{field_str}.json"
    if os.path.isfile(memoized_fn):
        with open(memoized_fn, 'r') as memoized_file:
            one_log_json = json.load(memoized_file)
        return one_log_json
    else:
        return []


def memoize_husky_data(one_field_list: List[Any], log_fn: str, field_type: str) -> None:
    field_str = topic_to_fn_topic(field_type)
    memoized_fn = f"{log_fn}_{field_str}.json"
    if os.path.isfile(memoized_fn):
        return
    else:
        with open(memoized_fn, 'w') as memoized_file:
            json.dump(one_field_list, memoized_file)


def convert_logs_husky(log_fns: List[Tuple[str, str, str]],
                       field_type: str="/ground_truth/state_map",
                       alt_bag_base = None) -> List[Tuple[str, str, str, np.array]]:
    print(f"convert_logs_husky")
    fn_count = 0
    one_field_lists = []
    total_fns = len(log_fns)
    #logging.debug(f"convert_logs_husky log_fns: {log_fns}")
    for log_fn, mutation_fn, mission_fn in log_fns:

        if alt_bag_base:
            print("changing bag base")
            log_fn = change_bag_base(log_fn, alt_bag_base)

        print(f"convert_logs_husky log_fn: {log_fn}")

        # try to get the memoized data
        husky_data = get_memoized_husky_data(log_fn, field_type)

        if len(husky_data) > 0:
            one_field_list = husky_data
        else:
            if not os.path.isfile(log_fn):
                print(f"file missing: {log_fn}\nContinuing")
                fn_count += 1
                continue

            try:
                bag = rosbag.Bag(log_fn)
            except rosbag.bag.ROSBagException as e:
                logging.info(f"file problem: {log_fn}\nContinuing")
                fn_count += 1
                continue
            bag_contents = bag.read_messages()
            bag_name = bag.filename

            logs_by_topic: Dict[str, Any] = dict()

            for topic, msg, t in bag_contents:
                #logging.debug(f"topic: {topic}, msg: {msg}, t: {t}")
                if topic in logs_by_topic:
                    logs_by_topic[topic] = logs_by_topic[topic] + [(msg, t)]
                else:
                    logs_by_topic[topic] = [(msg, t)]

            try:
                data = logs_by_topic[field_type]
            except KeyError as e:
                logging.warning(f"No field {field_type} in bag: {log_fn}.\nContinuing")
                fn_count += 1
                continue
            if field_type == "/ground_truth/state_map":
                one_field_list = [ (x[1].to_nsec(),
                                    x[0].pose.pose.position.x,
                                    x[0].pose.pose.position.y,
                                    x[0].pose.pose.position.z)
                                   for x in data]
        one_field_np = np.array(one_field_list)
        memoize_husky_data(one_field_list, log_fn, field_type)
        one_field_lists.append((log_fn, mutation_fn, mission_fn, one_field_np))
        fn_count += 1
        logging.debug(f"Finished file {fn_count} of {total_fns}")
    return one_field_lists


def convert_logs_ardu(log_fns: List[Tuple[str, str, str]],
                      field_type: str="GLOBAL_POSITION_INT") -> List[Tuple[str, str, str, np.array]]:
    one_field_lists = []
    fn_count = 0
    total_fns = len(log_fns)
    logging.debug(f"convert_logs_ardu log_fns: {log_fns}")
    for log_fn, mutation_fn, mission_fn in log_fns:
        logging.debug(f"convert_logs log_fn: {log_fn}")
        # convert the tlog to json, if it's not already json
        log_json = get_json(log_fn)

        one_field_dict = [x["data"] for x in log_json if
                          x["meta"]["type"] == field_type]
        timestamp_list = [x["meta"]["timestamp"] for x in log_json if
                          x["meta"]["type"] == field_type]
        assert(len(one_field_dict) == len(timestamp_list))

        one_field_list: List[Any] = []

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
        one_field_lists.append((log_fn, mutation_fn, mission_fn, one_field_np))
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


def logs_to_np(log_fns: Dict[str, List[Tuple[str, str, str]]], log_type="ardu",
               alt_bag_base=None) -> Dict[str, List[Tuple[str, str, str, Any]]]:
    all_logs = dict()
    # Turn the log files into np.arrays of the data
    for label, log_fns_subset in log_fns.items():
        logging.debug(f"label: {label}, log_fns_subset: {log_fns_subset}")
        if log_type == "ardu":
            logs_data = convert_logs_ardu(log_fns_subset)
        elif log_type == "husky":
            logs_data = convert_logs_husky(log_fns_subset, alt_bag_base=alt_bag_base)
        all_logs[label] = logs_data

    # logging.debug(f"all_logs: {all_logs}")
    return all_logs


def to_json_ready(all_logs: Dict[str, List[Tuple[str, str, str, np.array]]]) -> Dict[str, List[Tuple[str, str, str, List]]]:
    json_ready = dict()
    for key in all_logs:
        json_ready_list = []
        for item in all_logs[key]:
            json_ready_list.append((item[0], item[1], item[2], item[3].tolist()))
        json_ready[key] = json_ready_list
    return json_ready


def json_to_np(json_logs: Dict[str, List[Tuple[str, str, str, List]]]) -> Dict[str, List[Tuple[str, str, str, np.array]]]:
    all_logs = dict()
    for key in json_logs:
        all_logs_list = []
        for item in json_logs[key]:
            # logging.debug(f"item: {item}")
            all_logs_list.append((item[0], item[1], item[2], np.array(item[3])))
        all_logs[key] = all_logs_list
    return all_logs


def memoize_log_db(log_db: str, log_type="ardu",
                   alt_bag_base=None) -> Dict[str, List[Tuple[str, str, str, np.array]]]:
    date = time.strftime("%m-%d-%Y")
    memoized_fn = f"{log_db}_{date}.json"
    logging.debug(memoized_fn)

    if not os.path.isfile(memoized_fn):
        log_fns = get_from_db(log_db, log_type=log_type)
        all_logs = logs_to_np(log_fns, log_type=log_type,
                              alt_bag_base=alt_bag_base)
        with open(memoized_fn, 'w') as memoized_file:
            json_ready_logs = to_json_ready(all_logs)
            json.dump(json_ready_logs, memoized_file)
    else:
        with open(memoized_fn, 'r') as memoized_file:
            json_logs = json.load(memoized_file)
            all_logs = json_to_np(json_logs)
    return all_logs


def get_logs(args: argparse.Namespace) -> Dict[str, List[Tuple[str, str, str, np.array]]]:
    log_fns: Dict[str, List[Tuple[str, str, str]]] = dict()

    # if args.log_fn and len(args.log_fn) > 1:
    if args.bag_dir or (args.log_fn and len(args.log_fn) > 0):
        if args.bag_dir:
            log_fn_list = [x for x in os.listdir(args.bag_dir)
                           if x.endswith(".bag")]
            log_fn_list = [os.path.join(args.bag_dir, x) for x in log_fn_list]
        elif args.log_fn and len(args.log_fn) > 0:
            log_fn_list = args.log_fn
        logging.debug(f"log_fn_list: {log_fn_list}")

        log_fns = insert(log_fns, "manual", [(x, "None", "None") for x
                                             in log_fn_list] )
        logging.debug(f"log_fns: {log_fns}")
        all_logs = logs_to_np(log_fns, args.log_type)
    elif args.log_db:
        #TODO: This assumes all the logs in the database have the same
        # mission, which isn't always a good assumption. Fix it.
        all_logs = memoize_log_db(args.log_db, args.log_type,
                                  alt_bag_base=args.alt_bag_base)

    return all_logs


def logs_by_mission(logs: Dict[str, List[Tuple[str, str, str, np.array]]]) -> Dict[str, Dict[str, List[Tuple[str, str, np.array]]]]:
    by_mission: Dict[str, Dict[str, List[Tuple[str, str, np.array]]]] = dict()
    for label in logs.keys():
        logging.debug(f"logs_by_mission key: {label} length: {len(logs[label])}")
        if len(logs[label]) == 0:
            logging.debug(f"no logs for {label}")
        for log_fn, mutation_fn, mission_fn, log in logs[label]:
            if mission_fn not in by_mission:
                by_mission[mission_fn] = dict()
            if label in by_mission[mission_fn]:
                by_mission[mission_fn][label] = \
                    by_mission[mission_fn][label] + \
                    [(log_fn, mutation_fn, log)]
            else:
                by_mission[mission_fn][label] = [(log_fn, mutation_fn, log)]

    return by_mission


def insert(db: Dict[str, List[Any]], label: str,
           data: List[Any]) -> Dict[str, List[Any]]:
    #logging.debug(db)
    if label in db:
        db[label] = db[label] + data
    else:
        db[label] = data
    return db


def mission_to_list(mission_fn: str,
                    log_type: str = "ardu",
                    alt_mission_base: str = None) -> List[Tuple[float, float, float]]:

    if alt_mission_base:
        mission_fn = change_bag_base(mission_fn, alt_mission_base)


    try:
        with open(mission_fn, 'r') as y:
            data = yaml.load(y, Loader=yaml.BaseLoader)
    except FileNotFoundError:
        print(f"File not found: {mission_fn}")
        return []
    positions = [p["position"] for p in data["poses"]]
    pos_list = [(p['x'], p['y'], p['z']) for p in positions]
    return pos_list
