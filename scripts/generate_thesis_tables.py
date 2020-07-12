import argparse
import logging
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
        if label == "nominal":
            delay = 0
            topic = "None"
            log_data_list = [x[3] for x in
                             log_analysis.convert_logs_husky(
                    log_fn_list,
                    alt_bag_base = args.alt_bag_base)]
            runs_by_mission = dict({"None": {0.0: {"None": log_data_list}}})

        elif label == "experimental":
            mission_fns = set([x[2] for x in log_fn_list])
            logging.debug(f"mission_fns: {mission_fns}")
            for mission_fn in mission_fns:
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
    for mission_fn in experimental_runs_by_mission:
        logging.debug(f"mission_fn: {mission_fn}")
        mission_as_list = log_analysis.mission_to_list(mission_fn,
                                                       log_type = "husky",
                                                       alt_mission_base = args.alt_mission_base)
        # For each delay amount in the mission?? (separate this way??)
        for mission_fn, delay_amt_dict in experimental_runs_by_mission.items():
            # For each execution in the mission
            #print(f"type(delay_amt): {type(delay_amt)}, delay_amt: {delay_amt}")
            print(f"type(delay_amt_dict): {type(delay_amt_dict)}")
            print(f"len(delay_amt_dict): {len(delay_amt_dict)}")
            print(f"keys(delay_amt_dict): {keys(delay_amt_dict)}")
            for key in delay_amt_dict.keys():
                print(f"len(delay_amt_dict[{key}]: {len(delay_amt_dict[key])}")
            #for topic in delay_amt:

            #    for log in topic_dict:
            #        print(f"log: {log}")
                    # Calculate the minimum distance to each waypoint
            #        waypoint_dists = log_analysis.distance_to_each_waypoint(log,
#                                                                            mission_as_list,
 #                                                                           log_type="husky")

                    #if len(waypoint_dists) > 0:
                        #print(f"waypoint_dists: {waypoint_dists}")

                # Find the final distances (maybe the same as the last waypoint)
                # Calculate the mean over all the waypoints
        pass


 # Find the mean of the minimum distances for each waypoint
    # Find the standard deviation for the minimum distances for each WP

    pass


def main():
    args = parse_args()

    if args.graph == "waypoint_distance":
        waypoint_distance(args)


if __name__ == '__main__':
    main()
