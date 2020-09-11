import argparse
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple

import docker
import roswire
from roswire.definitions import FormatDatabase, TypeDatabase
from dockerblade import Shell as ROSWireShell


DIR_THIS = os.path.dirname(os.path.abspath(__file__))

bag_dir = '../bags/fetch/'



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_fn', type=str, default="fetch_exp.log")
    parser.add_argument('--bag_fn', type=str, default="fetch.bag")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(filename=args.log_fn, level=logging.INFO,
                        format=format_str, datefmt=date_str)

    bag_fn = args.bag_fn
    bag_dir_abs = os.path.join(DIR_THIS, bag_dir)
    if not os.path.isdir(bag_dir_abs):
        os.makedirs(bag_dir_abs)

    bagfile = os.path.join(bag_dir_abs, bag_fn)
    assert(os.path.isdir(os.path.dirname(bagfile)))

    rsw = roswire.ROSWire()
    sources = ['/opt/ros/melodic/setup.bash', '/ros_ws/devel/setup.bash']
    logging.info("Launching system")
    with rsw.launch('fetch:headless', sources) as system:
        with system.roscore() as ros:
            logging.info("Launching playground environment")
            ros.roslaunch('pickplace_playground.launch', package='fetch_gazebo')
            logging.info("Sleeping for 60 seconds")
            time.sleep(60)
            logging.info("Launching pick_place demo")
            ros.roslaunch('pick_place_demo.launch', package='fetch_gazebo_demo')
            # TODO
            # Figure out roughly how long this takes and sleep for just a
            # little longer

            # Hardcoded topics to exclude
            to_exclude = '/clock|/gazebo/model_states'

            with ros.record(bagfile, exclude_topics=to_exclude) as recorder:
                logging.info("Sleeping for 60 seconds")
                time.sleep(60)


if __name__ == "__main__":
    main()
