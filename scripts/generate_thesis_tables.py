import argparse
import logging
import statistics
import sys
from typing import Dict, List, Tuple

import numpy as np

import log_analysis

mission_num_to_fn = {
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
    args = parser.parse_args()
    return args


def print_table(dict_by_label_mission, graph_type):
    if graph_type == "waypoint_distance":
        for label, mission_fn_dict in dict_by_label_mission.items():
            print(f"label: {label}")
            for i in range(1, len(mission_num_to_fn)):
                str_to_print = f"M{i} & "
                value_dict = [x[1] for x in mission_fn_dict.items() if
                              mission_num_to_fn[i] in x[0]]
                if len(value_dict) == 0:
                    continue
                value_dict = value_dict[0]
                for j in range(len(value_dict)):
                    str_to_print += f"{value_dict[j]:.2f} & "
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


def waypoint_distance_delay(args):

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

    delay_dict = dict()
    for bag_fn, topic_delay_fn, mission_fn, log_data in bag_data_one_mission:
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
            log_type=args.log_type)
        if delay not in delay_dict:
            delay_dict[delay] = []
        delay_dict[delay] = delay_dict[delay] + [waypoint_dict]

    mean_delay_dict = dict()
    std_delay_dict = dict()

    for delay, delay_list in delay_dict.items():
        logging.debug(f"delay: {delay} delay_list: {delay_list}")
        num_data_points = len(delay_list)
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

    print("Mean")
    print_table(mean_delay_dict, args.graph)
    print("Standard Deviation")
    print_table(std_delay_dict, args.graph)
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


def waypoint_distance_old(args):
    # Get the set of experimental runs by mission
    experimental_runs_by_mission = get_experimental_runs_by_mission(args)
    # For each mission
    num_missions = len(experimental_runs_by_mission)
    mission_count = 0
    num_waypoints = dict()

    for mission_fn, delay_amt_dict in experimental_runs_by_mission.items():
        mission_count += 1
        logging.debug(f"mission {mission_count} of {num_missions}\nmission_fn: {mission_fn}")
        mission_as_list = log_analysis.mission_to_list(mission_fn,
                                                       log_type = "husky",
                                                       alt_mission_base = args.alt_mission_base)
        num_waypoints[mission_fn] = len(mission_as_list)
        waypoint_dist_dict = dict()
        # For each execution in the mission
        #print(f"type(delay_amt): {type(delay_amt)}, delay_amt: {delay_amt}")
        mission_dist_list = []

        num_delays = len(delay_amt_dict)
        delay_count = 0
        for delay_amt, topics_dict in delay_amt_dict.items():
            delay_count += 1
            logging.debug(f"delay {delay_count} of {num_delays}; delay_amt: {delay_amt}\n(mission {mission_count} of {num_missions})")
            for topic, log_list in topics_dict.items():
                logging.debug(f"len(log_list): {len(log_list)}")
                num_logs = len(log_list)
                log_count = 0
                for log in log_list:
                    log_count += 1
                    logging.debug(f"log {log_count} of {num_logs}\ndelay {delay_count} of {num_delays}; (mission {mission_count} of {num_missions})")
                    # Calculate the minimum distance to each waypoint
                    waypoint_dists = log_analysis.distance_to_each_waypoint(
                        log,
                        mission_as_list,
                        log_type="husky")

                    num_wp = num_waypoints[mission_fn]
                    waypoint_dists["num_waypoints"] = num_wp
                    waypoint_dists["delay_amt"] = float(delay_amt)

                    # Calculate the mean over all the waypoints
                    one_mean = statistics.mean(waypoint_dists.values())
                    waypoint_dists["mean"] = one_mean
                    mission_dist_list.append(waypoint_dists)
        waypoint_dist_dict[mission_fn] = mission_dist_list


    #with open("waypoint_dist_dict.json", "w") as w:
    #    try:
    #        json.dump(waypoint_dist_dict, w)
    #    except:
    #        logging.error("JSON dump failed")
    #        pass

    # Find the mean of the minimum distances for each waypoint over all experimental
    logging.debug(len(waypoint_dist_dict))
    for mission_fn, waypoint_dist_list in waypoint_dist_dict.items():
        dist_list = dict()
        dist_list["nominal"] = [x for x in waypoint_dist_list if
                                x["delay_amt"] == 0]
        dist_list["experimental"] = [x for x in waypoint_dist_list if
                                     x["delay_amt"] > 0]
        print(f"mission_fn: {mission_fn}")
        for i in range(num_waypoints[mission_fn]):
            for group in ["experimental", "nominal"]:
                if len(dist_list[group]) == 0:
                    logging.debug(f"dist_list[{group}] is zero. contiuing")
                    continue
                mission_means = statistics.mean([x[i] for x in
                                                 dist_list[group]])
                print(f"means {group}")
                print(f"{mission_means[0]} & {mission_means[1]} & {mission_means[2]} & {mission_means[3]} & {mission_means[4]} & {mission_means[5]}")
       # Find the standard deviation for the minimum distances for each WP
                mission_stdev = statistics.stdev([x[i] for x in
                                                  dist_list[group]])
                print(f"stdev {group}")
                print(f"{mission_stdev[0]} & {mission_stdev[1]} & {mission_stdev[2]} & {mission_stdev[3]} & {mission_stdev[4]} & {mission_stdev[5]}")



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
        bag_data = log_analysis.convert_logs_husky(
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
                log_type="husky",
                alt_mission_base=args.alt_mission_base)
            mission_data = [x for x in bag_data if x[2] == mission_fn]
            #logging.debug(f"*"*80)
            #logging.debug(f"\n\nlabel: {label}, mission: {mission_fn}:\n {mission_data}")

            dists_list = []

            for bag_fn, delay_fn, mission_fn_loc, log_data in mission_data:
                assert(mission_fn.strip() == mission_fn_loc.strip()), f"{mission_fn} {mission_fn_loc}"
                dists = log_analysis.distance_to_each_waypoint(log_data,
                                                               mission_as_list,
                                                               log_type="husky")
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

    if args.graph == "waypoint_distance":
        print(f"delay_dict: {delay_dict}")
        for delay, delay_lists in delay_dict:
            print(f"delay: {delay}")
            print(f"delay_dict: {delay_dict}")
        return
        print_table(mean_delay_dict, args.graph)
        print_table(std_delay_dict, args.graph)

        print("Mean")
        print_table(mean_dist_dict_delay, args.graph)
        print("Standard Deviation")
        print_table(std_dict_delay, args.graph)


def waypoint_distance_by_topic(args):

    bag_fns = log_analysis.get_from_db(args.log_db, log_type=args.log_type)
    all_bag_fns = bag_fns["nominal"] + bag_fns["experimental"]




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


if __name__ == '__main__':
    main()
