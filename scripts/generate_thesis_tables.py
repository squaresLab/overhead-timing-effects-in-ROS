import argparse
import logging
import statistics
import sys
from typing import Dict, List, Tuple

import numpy as np

import log_analysis

mission_num_to_fn_husky = {
1: "HUSKY_pose_array_5_156109eab4cc4c1f9432690d0b6e6ca9.yaml",
2: "HUSKY_pose_array_5_c8629fc042474a9db240282a52448964.yaml",
3: "HUSKY_pose_array_5_5f31cf0a67be40a4b9adcb331f43361c.yaml",
4: "HUSKY_pose_array_5_350028c9199b4d15b236d85c10ff8b9e.yaml",
5: "HUSKY_pose_array_5_2ce0b4fa2a4249b4b5057913b8c3d9ec.yaml",
6: "HUSKY_pose_array_5_c787950abab14110836ae170213406d8.yaml",
7: "HUSKY_pose_array_5_865a8b7817474e51861522bd435faff1.yaml",
8: "HUSKY_pose_array_5_722ba706e1f34beca0bcaf193d7e0c4f.yaml",
9: "HUSKY_pose_array_5_cdc8edbd1c2b4ce197a211ee64e7cbe4.yaml",
10: "HUSKY_pose_array_5_de68abeac0924afc938d17e5acd1b825.yaml",
11: "HUSKY_pose_array_5_a4d27db27974459a97a696b7d98cb403.yaml"}

mission_num_to_fn_ardu = {
1: 'missions/auto/3f9c851f75a4421690482d330d496e09.wpl',
2: 'missions/auto/42adb8a3b23c41649b348767b1372095.wpl',
3: 'missions/auto/5839c2a439e64653bbf8d75088f2b11e.wpl',
4: 'missions/auto/70f96678dda24bf9b5520c0c1b9b13d9.wpl',
5: 'missions/auto/a33b9119fd0d4bd0a307bd0b516aa0fe.wpl',
6: 'missions/auto/c288ab429d1d4b8a813ce76612c74764.wpl',
7: 'missions/auto/ca9ce16f7a044203926a69d82f63b048.wpl',
8: 'missions/auto/d009e94f7c044147973d129f33d4d0a3.wpl',
9: 'missions/auto/d7cbe848579a44879651444bf3c29900.wpl',
10: 'missions/auto/f3a59206b2674e14b50d3bfb2b5ea63b.wpl',
11: 'missions/auto/f57767682748498caee204d6903018b0.wpl'}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str,
                        help="which graph to generate")
    parser.add_argument("--log_db", type=str)
    parser.add_argument("--log_type", type=str, default="ardu")
    parser.add_argument("--alt_bag_base", type=str,
                        help="specify where the bag files referenced in the log_db reside")
    parser.add_argument("--alt_mission_base", type=str)
    parser.add_argument("--logging_fn", type=str, default="generate_tables.log")
    parser.add_argument("--topic", type=str, default="/husky_velocity_controller/odom")
    parser.add_argument("--one_mission", type=str)
    parser.add_argument("--delay", type=float)
    parser.add_argument("--threshold", type=float,
                        help="how close to a WP the robot needs to get to be considered there")
    parser.add_argument("--final_distance", action="store_true",
                        default=False,
                        help="When computing closest distances of a log to the waypoints, use the final position of the robot for the final waypoint, instead of the closest")
    args = parser.parse_args()
    return args


def print_table(dict_by_label_mission, graph_type, log_type):
    if graph_type == "waypoint_distance":
        #logging.debug(f"dict_by_label_mission: {dict_by_label_mission}")
        for label, mission_fn_dict in dict_by_label_mission.items():
            print(f"{label} \\\\")
            #logging.debug(f"mission_fn_dict: {mission_fn_dict}")
            if log_type == "husky":
                mission_num_to_fn = mission_num_to_fn_husky
            elif log_type == "ardu":
                mission_num_to_fn = mission_num_to_fn_ardu
            for i in range(1, len(mission_num_to_fn)):
                str_to_print = f"M{i} & "
                #logging.debug(f"mission_fn_dict: {mission_fn_dict}")
                value_dict = [x[1] for x in mission_fn_dict.items() if
                              mission_num_to_fn[i] in x[0]]
                #logging.debug(f"value_dict: {value_dict}")
                if len(value_dict) == 0:
                    continue
                value_dict = value_dict[0]
                for j in value_dict.values():
                    try:
                        str_to_print += f"{j:.2f} & "
                    except KeyError as e:
                        print(f"KeyError: {e}\nkey: {key}")
                        str_to_print += f" & "
                sum_values = sum(value_dict.values())
                str_to_print += f"{sum_values:.2f} & "
                mean_values = statistics.mean(value_dict.values())
                str_to_print += f"{mean_values:.2f} \\\\"
                print(str_to_print)

    if graph_type == "waypoint_distance_delay":
        dict_by_label_mission_items = sorted(dict_by_label_mission.items(),
                                             key=lambda x: x[0])
        for delay, waypoint_dict in dict_by_label_mission_items:
            str_to_print = f"{delay} & "
            value_dict = waypoint_dict
            if len(value_dict) == 0:
                continue
            waypoint_numbers = sorted(waypoint_dict.keys())

            for j in waypoint_numbers:
                if j in value_dict:
                    str_to_print += f"{value_dict[j]:.2f} & "
                else:
                    str_to_print += f" & "
            sum_values = sum(value_dict.values())
            str_to_print += f"{sum_values:.2f} & "
            mean_values = statistics.mean(value_dict.values())
            str_to_print += f"{mean_values:.2f} \\\\"
            print(str_to_print)


def escape_underscores(in_str: str) -> str:
    if "_" not in in_str:
        return in_str

    out_str = in_str.replace("_", "\\_")
    return out_str


def waypoint_distance_delay(args):

    assert(args.one_mission)

    bag_fns = log_analysis.get_from_db(args.log_db, log_type=args.log_type)
    all_bag_fns_list = bag_fns["nominal"] + bag_fns["experimental"]

    all_mission_fns = set([x[2] for x in all_bag_fns_list])
    logging.debug(f"all_mission_fns: {all_mission_fns}")

    one_mission_fns = [x for x in all_mission_fns if args.one_mission in x]
    assert(len(one_mission_fns) <= 1)

    bag_fns_one_mission = [x for x in all_bag_fns_list if x[2] ==
                               one_mission_fns[0]]
    if args.log_type == "husky":
        bag_data_one_mission = log_analysis.convert_logs_husky(
            bag_fns_one_mission,
            alt_bag_base = args.alt_bag_base)

    elif args.log_type == "ardu":
        bag_data_one_mission = log_analysis.convert_logs_ardu(
            bag_fns_one_mission,
            alt_bag_base = args.alt_bag_base)

    mut_fns = [x[1] for x in bag_data_one_mission]
    logging.debug(f"mut_fns: {mut_fns}")

    delay_dict = dict()
    for bag_fn, topic_delay_fn, mission_fn, log_data in bag_data_one_mission:
        if len(log_data) == 0:
            logging.warn(f"no data for {bag_fn}")
            continue
        topic, delay = log_analysis.mut_fn_to_topic_delay(
            topic_delay_fn,
            log_type=args.log_type)
        mission_as_list = log_analysis.mission_to_list(
            mission_fn,
            log_type=args.log_type,
            alt_mission_base=args.alt_mission_base)
        waypoint_dict = log_analysis.distance_to_each_waypoint(
            log_data,
            mission_as_list,
            log_type=args.log_type,
            final_dist=args.final_distance)
        if delay not in delay_dict:
            delay_dict[delay] = []
        delay_dict[delay] = delay_dict[delay] + [waypoint_dict]

    mean_delay_dict = dict()
    std_delay_dict = dict()

    for delay, delay_list in delay_dict.items():
        logging.debug(f"delay: {delay} delay_list: {delay_list}")
        num_data_points = len(delay_list)
        if num_data_points < 2:
            continue
        num_waypoints = len(delay_list[0])
        logging.debug(f"delay: {delay} num_data_points: {num_data_points} num_waypoints: {num_waypoints}")

        for waypoint in sorted(delay_list[0].keys()):
            vals = [x[waypoint] for x in delay_list]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            if delay not in mean_delay_dict:
                mean_delay_dict[delay] = dict()
            if delay not in std_delay_dict:
                std_delay_dict[delay] = dict()

            mean_delay_dict[delay][waypoint] = mean
            std_delay_dict[delay][waypoint] = std

    print("Mean \\\\")
    print_table(mean_delay_dict, args.graph, args.log_type)
    print("Standard Deviation \\\\")
    print_table(std_delay_dict, args.graph, args.log_type)
    return


def topic_in_fn(fn: str, topic: str) -> bool:
    topic_words: List[str] = []
    topic_short = topic.strip("_/")
    topic_split_slash = topic.split("/")
    for word in topic_split_slash:
        topic_words.extend(word.split("_"))
    #print(topic_words)
    #print(fn)
    #print(topic)
    all_in = all([x in fn for x in topic_words])
    #print(all_in)
    return all_in


def get_delay_topic_dict(log_fn_list: List[Tuple[str, str, str]],
                   delay: float = 0.0, alt_bag_base: str = None,
                   log_type: str = "ardu") -> Dict[float, Dict[str, List[np.array]]]:

    mut_fns = [x[1] for x in log_fn_list]
    logging.debug(f"mut_fns: {mut_fns}")

    topics_dict: Dict[str, List[np.array]] = dict()
    topics = set([log_analysis.mut_fn_to_topic_delay(x[1], log_type=log_type)[0]
                  for x in log_fn_list])
    delays = set([log_analysis.mut_fn_to_topic_delay(x[1], log_type=log_type)[1]
                  for x in log_fn_list])
    print(f"topics: {topics}")

    delay_topics_dict: Dict[float, Dict[str, List[np.array]]] = dict()

    for delay in delays:

        print(delay)
        print(len(log_fn_list))
        one_delay_list = [x for x in log_fn_list if
                          delay in x]
        print(len(one_delay_list))
        for topic in topics:
            one_topic_fn_list = [x for x in one_delay_list if
                                 topic_in_fn(x[1],topic)] # and
                                 #str(delay) in x]
            print(f"one_topic_fn_list: {one_topic_fn_list}")
            if log_type == "husky":
                one_topic_data_list = \
                    log_analysis.convert_logs_husky(one_topic_fn_list,
                                                    alt_bag_base = alt_bag_base)

            if log_type == "ardu":
                raise NotImplementedError

            assert(len(one_topic_fn_list) == len(one_topic_data_list)), f"len_one_topic_fn_list: {len(one_topic_fn_list)}, len(one_topic_data_list): {len(one_topic_data_list)}"
            logging.debug(f"len_one_topic_fn_list: {len(one_topic_fn_list)}, len(one_topic_data_list): {len(one_topic_data_list)}")
            topics_dict[topic] = one_topic_data_list
        delay_topics_dict[delay] = topics_dict

    print(f"delay_topics_dict: {delay_topics_dict}")
    return delay_topics_dict


def get_experimental_runs_by_mission(args) -> Dict[str, Dict[float, Dict[str, List[np.array]]]]:
    runs_by_mission: Dict[str, Dict[float, Dict[str, List[np.array]]]] = dict()
    log_fns = log_analysis.get_from_db(args.log_db, log_type=args.log_type)
    for label, log_fn_list in log_fns.items():
        mission_fns = set([x[2] for x in log_fn_list])
        for mission_fn in mission_fns:
            if label == "nominal":
                delay = 0
                topic = "None"
                log_data_list = [x[3] for x in
                                 log_analysis.convert_logs_husky(
                        log_fn_list,
                        alt_bag_base = args.alt_bag_base)]
                delay_topic_dict = dict({0.0: {"None": log_data_list}})

                runs_by_mission[mission_fn] = delay_topic_dict

            elif label == "experimental":
                logging.debug(f"mission_fns: {mission_fns}")
                delay_topic_dict = get_delay_topic_dict(
                    log_fn_list, delay=delay,
                    log_type=args.log_type, alt_bag_base=args.alt_bag_base)
                    #nominal_dict[mission_fn] = dict({delay: topic_dict})
                runs_by_mission[mission_fn] = delay_topic_dict
    return runs_by_mission


def waypoint_distance(args):
    # get the appropriate group of data. do this only once.
    bag_fns = log_analysis.get_from_db(args.log_db, log_type=args.log_type)

    all_bag_fns_list = bag_fns["nominal"] + bag_fns["experimental"]

    mission_fns = set([x[2] for x in all_bag_fns_list])
    logging.debug(f"mission_fns: {mission_fns}")

    husky_bag_data: Dict[str, Tuple[str, str, np.array]] = dict()

    dist_by_label_mission: Dict[str, Dict[str, List[Dict[int, float]]]] = dict()
    mean_by_label_mission: Dict[str, Dict[str, Dict[int, float]]] = dict()
    std_by_label_mission: Dict[str, Dict[str, Dict[int, float]]] = dict()

    for label in ("nominal", "experimental"):

        # bag_data is a list of Tuples.
        # Each tuple is bag_fn, delay_fn, mission_fn, log_data
        # log_data is an np.array of [time_elapsed, x_pos, y_pos, z_pos
        if args.log_type == "husky":
            bag_data = log_analysis.convert_logs_husky(
                bag_fns[label],
                alt_bag_base = args.alt_bag_base)
        elif args.log_type == "ardu":
            bag_data = log_analysis.convert_logs_ardu(
                bag_fns[label],
                alt_bag_base = args.alt_bag_base)
        husky_bag_data[label] = bag_data

        if args.one_mission:
            mission_fns = [x for x in mission_fns if args.one_mission in x]
            print(mission_fns)

        # Separate by mission
        for mission_fn in mission_fns:
            mission_as_list = log_analysis.mission_to_list(
                mission_fn,
                log_type=args.log_type,
                alt_mission_base=args.alt_mission_base)
            mission_data = [x for x in bag_data if x[2] == mission_fn]
            #logging.debug(f"*"*80)
            #logging.debug(f"\n\nlabel: {label}, mission: {mission_fn}:\n {mission_data}")

            dists_list = []

            for bag_fn, delay_fn, mission_fn_loc, log_data in mission_data:
                assert(mission_fn.strip() == mission_fn_loc.strip()), f"{mission_fn} {mission_fn_loc}"
                if len(log_data) == 0:
                    logging.error(f"log data missing for bag: {bag_fn}")
                    continue
                dists = log_analysis.distance_to_each_waypoint(
                    log_data,
                    mission_as_list,
                    log_type=args.log_type,
                    final_dist=args.final_distance)
                dists_list.append(dists)

            if label not in dist_by_label_mission:
                dist_by_label_mission[label] = dict()
            dist_by_label_mission[label][mission_fn] = dists_list

            #logging.debug(f"*"*80)
            #logging.debug(f"\n\ndist_by_label_mission[{label}][{mission_fn}]: {dist_by_label_mission[label][mission_fn]}")

            mean_dist_dict = dict()
            std_dict = dict()

            if args.graph == "waypoint_distance":
                for waypoint in dists_list[0].keys():
                    vals = [x[waypoint] for x in dists_list]
                    mean = statistics.mean(vals)
                    mean_dist_dict[waypoint] = mean
                    std = statistics.stdev(vals)
                    std_dict[waypoint] = std
                if label not in mean_by_label_mission:
                    mean_by_label_mission[label] = dict()
                mean_by_label_mission[label][mission_fn] = mean_dist_dict
                if label not in std_by_label_mission:
                    std_by_label_mission[label] = dict()
                std_by_label_mission[label] = std_dict

    print("Mean \\\\")
    print_table(mean_by_label_mission, args.graph, args.log_type)
    logging.debug("Std deviation doesn't work yet")
    # print("Standard Deviation \\\\")
    # print_table(std_by_label_mission, args.graph, args.log_type)


def waypoint_distance_by_topic(args):
    assert(type(args.delay) == float), args.delay
    assert(args.one_mission), args.one_mission

    bag_fns = log_analysis.get_from_db(args.log_db, log_type=args.log_type)
    all_bag_fns_list = bag_fns["nominal"] + bag_fns["experimental"]
    logging.debug(f"all_bag_fns_list: {all_bag_fns_list}")

    all_mission_fns = set([x[2] for x in all_bag_fns_list])

    one_mission_fns = [x for x in all_mission_fns if args.one_mission in x]
    assert(len(one_mission_fns) == 1)

    bag_fns_one_mission = [x for x in all_bag_fns_list if x[2] ==
                               one_mission_fns[0]]

    bag_fns_one_mission_one_delay = [
        x for x in bag_fns_one_mission if
        (args.delay ==
         log_analysis.mut_fn_to_topic_delay(x[1], log_type="husky")[1])]

    logging.debug(f"len(bag_fns_one_mission_one_delay): {len(bag_fns_one_mission_one_delay)}")

    bag_data_one_mission_one_delay = log_analysis.convert_logs_husky(
        bag_fns_one_mission_one_delay,
        alt_bag_base = args.alt_bag_base)


    topic_dict = dict()
    for bag_fn, topic_delay_fn, mission_fn, log_data \
            in bag_data_one_mission_one_delay:
        topic, delay = log_analysis.mut_fn_to_topic_delay(
            topic_delay_fn,
            log_type=args.log_type)
        assert(delay == args.delay), f"delay_from_fn: {delay}, args.delay: {args.delay}\ntopic_delay_fn: {topic_delay_fn}"
        mission_as_list = log_analysis.mission_to_list(
            mission_fn,
            log_type=args.log_type,
            alt_mission_base=args.alt_mission_base)
        waypoint_dict = log_analysis.distance_to_each_waypoint(
            log_data,
            mission_as_list,
            log_type=args.log_type,
            final_dist=args.final_distance)
        if topic not in topic_dict:
            topic_dict[topic] = []
        topic_dict[topic] = topic_dict[topic] + [waypoint_dict]

    mean_topic_dict = dict()
    std_topic_dict = dict()

    for topic, topic_list in topic_dict.items():
        logging.debug(f"topic: {topic} topic_list: {topic_list}")
        num_data_points = len(topic_list)
        num_waypoints = len(topic_list[0])
        logging.debug(f"topic: {topic} num_data_points: {num_data_points} num_waypoints: {num_waypoints}")

        for waypoint in sorted(topic_list[0].keys()):
            vals = [x[waypoint] for x in topic_list]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            if topic not in mean_topic_dict:
                mean_topic_dict[topic] = dict()
            if topic not in std_topic_dict:
                std_topic_dict[topic] = dict()

            mean_topic_dict[topic][waypoint] = mean
            std_topic_dict[topic][waypoint] = std

    print("Mean \\\\")
    print_table(mean_topic_dict, "waypoint_distance_delay", args.log_type)
    print("Standard Deviation \\\\")
    print_table(std_topic_dict, "waypoint_distance_delay", args.log_type)
    return


def crashes(args):
    topic_dict, time_dict = get_topic_results_dict(args)

    all_delays = set()
    for topic, delay_dict in topic_dict.items():
        for delay in delay_dict.keys():
            all_delays.add(delay)

    for topic, delay_dict in sorted(topic_dict.items()):
        if topic == "None":
            continue
        to_print = f"{escape_underscores(topic)} & "
        for delay in sorted(all_delays):
            try:
                results_dict = delay_dict[delay]
            except KeyError:
                to_print += f" & "
                continue
            try:
                num_pass = results_dict["pass"]
                num_fail = results_dict["fail"]
            except KeyError as e:
                print(e)
                print(results_dict)
            percent = 100 * (num_pass / (num_pass + num_fail))
            to_print += f"{percent:.2f} & "
        to_print += "\\\\"
        print(to_print)


def get_topic_results_dict(args):
    bag_fns = log_analysis.get_from_db(args.log_db, log_type=args.log_type)
    all_bag_fns_list = bag_fns["nominal"] + bag_fns["experimental"]

    all_mission_fns = set([x[2] for x in all_bag_fns_list])

    one_mission_fns = [x for x in all_mission_fns if args.one_mission in x]
    assert(len(one_mission_fns) <= 1)

    bag_fns_one_mission = [x for x in all_bag_fns_list if x[2] ==
                               one_mission_fns[0]]

    bag_data_one_mission = log_analysis.convert_logs_husky(
        bag_fns_one_mission,
        alt_bag_base = args.alt_bag_base)

    topic_dict = dict()
    time_dict = dict()
    for bag_fn, topic_delay_fn, mission_fn, log_data in bag_data_one_mission:

        completion_time = log_analysis.completion_time(
            log_data, log_type="husky")

        topic, delay = log_analysis.mut_fn_to_topic_delay(
            topic_delay_fn,
            log_type=args.log_type)

        mission = log_analysis.mission_to_list(
            mission_fn,
            log_type=args.log_type,
            alt_mission_base=args.alt_mission_base)

        dist_dict = log_analysis.distance_to_each_waypoint(
            log_data, mission, log_type="husky",
            final_dist=args.final_distance)

        result_dict = log_analysis.reaches_waypoints(
            dist_dict,
            mission,
            log_type="husky",
            tolerance=args.threshold)
        result = all([x for x in result_dict.values()])

        if topic == None or topic == "None":
            assert(delay == 0), delay

        if topic not in topic_dict:
            topic_dict[topic] = dict({delay: {"pass": 0, "fail": 0}})
        if delay not in topic_dict[topic]:
            topic_dict[topic][delay] = dict({"pass": 0, "fail": 0})
        if topic not in time_dict:
            time_dict[topic] = dict({delay: {"pass": [], "fail": []}})
        if delay not in time_dict[topic]:
            time_dict[topic][delay] = dict({"pass": [], "fail": []})

        if result:
            topic_dict[topic][delay]["pass"] = \
                1 + topic_dict[topic][delay]["pass"]
            time_dict[topic][delay]["pass"] = \
                [completion_time] + time_dict[topic][delay]["pass"]
        else:
            topic_dict[topic][delay]["fail"] = \
                1 + topic_dict[topic][delay]["fail"]
            time_dict[topic][delay]["fail"] = \
                [completion_time] + time_dict[topic][delay]["fail"]

    logging.debug(f"topic_dict: {topic_dict}")
    logging.debug(f"time_dict: {time_dict}")

    assert(topic_dict.keys() == time_dict.keys())

    for topic in topic_dict.keys():
        topic_dict[topic][0.0] = {"pass": topic_dict["None"][0.0]["pass"],
                                  "fail": topic_dict["None"][0.0]["fail"]}
        topic_dict[topic][0.0]["pass"] = topic_dict["None"][0.0]["pass"]
        topic_dict[topic][0.0]["fail"] = topic_dict["None"][0.0]["fail"]

        time_dict[topic][0.0] = {"pass": time_dict["None"][0.0]["pass"],
                                 "fail": time_dict["None"][0.0]["fail"]}

    return topic_dict, time_dict


def crashes_time(args: argparse.Namespace):
    _, time_dict = get_topic_results_dict(args)

    all_delays = set()
    for topic, delay_dict in time_dict.items():
        for delay in delay_dict.keys():
            all_delays.add(delay)

    for topic, delay_dict in sorted(time_dict.items()):
        #logging.debug(f"delay_dict: {delay_dict}")
        if topic == "None":
            to_print = f"No Inserted Delay & "
        else:
            to_print = f"{escape_underscores(topic)} & "
        for delay in sorted(all_delays):
            if delay == 0.0:
                continue
            try:
                results_dict = time_dict[topic][delay]
            except KeyError:
                #logging.debug(f"KeyError on {delay}")
                to_print += f" & "
                continue
            try:
                pass_times = results_dict["pass"]
                fail_times = results_dict["fail"]
            except KeyError as e:
                print(e)
                print(results_dict)
            #logging.debug(pass_times)
            #logging.debug(fail_times)
            try:
                pass_mean = (statistics.mean(pass_times)) * 1e-9
                to_print += f" {pass_mean/60:.2f} & "
            except statistics.StatisticsError:
                pass_mean = ""
            try:
                fail_mean = statistics.mean(fail_times) * 1e-9
                to_print += f" {fail_mean/60:.2f} & "
            except statistics.StatisticsError:
                fail_mean = ""

        to_print += "\\\\"
        print(to_print)


def main():
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=args.logging_fn)
    ch = logging.StreamHandler()
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = "%m/%d/%Y %I:%M:%S %p"
    fh_formatter = logging.Formatter(fmt=format_str, datefmt=date_str)
    fh.setFormatter(fh_formatter)
    ch.setFormatter(fh_formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


    if "waypoint_distance" == args.graph:
        waypoint_distance(args)
    elif "waypoint_distance_delay" == args.graph:
        waypoint_distance_delay(args)
    elif "waypoint_distance_by_topic" == args.graph:
        waypoint_distance_by_topic(args)
    elif "crashes" == args.graph:
        crashes(args)
    elif "crashes_time" == args.graph:
        crashes_time(args)


if __name__ == '__main__':
    main()
