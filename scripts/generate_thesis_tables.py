import argparse
import logging
import statistics
import sys
from typing import Dict, List, Tuple

import numpy as np

import log_analysis


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
    args = parser.parse_args()
    return args


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
            #print(f"one_topic_fn_list: {one_topic_fn_list}")
            if log_type == "husky":
                one_topic_data_list = \
                    log_analysis.convert_logs_husky(one_topic_fn_list,
                                                    alt_bag_base = alt_bag_base)

            if log_type == "ardu":
                raise NotImplementedError

            assert(len(one_topic_fn_list) == len(one_topic_data_list)), f"len_one_topic_fn_list: {len(one_topic_fn_list)}, len(one_topic_data_list): {len(one_topic_data_list)}"
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
                print(f"stddev {group}")
                print(f"{mission_stdev[0]} & {mission_stdev[1]} & {mission_stdev[2]} & {mission_stdev[3]} & {mission_stdev[4]} & {mission_stdev[5]}")





    pass


def main():
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=args.logging_fn)
    ch = logging.StreamHandler()
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = "%m/%d/%Y %I:%M%S %p"
    fh_formatter = logging.Formatter(fmt=format_str, datefmt=date_str)
    fh.setFormatter(fh_formatter)
    ch.setFormatter(fh_formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


    if args.graph == "waypoint_distance":
        waypoint_distance(args)


if __name__ == '__main__':
    main()
