import argparse
import logging
import os
import shlex
import sqlite3
import subprocess
import time
from typing import List, Dict, Any, Optional, Tuple
import uuid
import yaml

import roswire
#from roswire.definitions import FormatDatabase, TypeDatabase
#from dockerblade import Shell as ROSWireShell

from experiment_runner import access_bag_db, file_hash, get_bag_fn

DIR_THIS = os.path.dirname(os.path.abspath(__file__))
bag_dir = '../bags/'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosrunner_yaml", type=str, action='append')
    # default="/usr0/home/dskatz/Documents/overhead-timing-effects-in-ROS/ROSRunner/husky_waypoints.yml")
    parser.add_argument("--db_fn", type=str, default="ros_bag_db.db")
    parser.add_argument("--log_fn", type=str, default="ros_experiment.log")
    parser.add_argument("--baseline_iterations", type=int, default=1)
    args = parser.parse_args()
    return args


def store_bag_fn_ros(*, bag_fn, docker_image_sha, docker_image,
                        container_uuid, mission_sha, mission_fn,
                        mutation_sha, mutation_fn, cursor, conn,
                        context) -> None:
    command = "INSERT INTO bagfns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    values = (bag_fn, docker_image_sha, docker_image, container_uuid,
              mission_sha, mission_fn, mutation_sha, mutation_fn, context)
    cursor.execute(command, values)
    conn.commit()

def run_one_experiment(sources: List[str], cursor, conn,
                       outbag: str="husky_waypoints.bag",
                       timeout: int=2000,
                       topic_regex: str="((.*)/move_base/(result|status|parameter(.*)|goal(.*))|(.*)/amcl(.*))",
                       param_fn: str="/usr0/home/dskatz/Documents/overhead-timing-effects-in-ROS/ROSRunner/husky_waypoints.yml",
                       docker_image_sha: str="None",
                       docker_image: str="None",
                       delay_fn: str="None", delay_sha: str="None",
                       mission_fn: str="None", mission_sha: str="None"):

    store_bag_fn_ros(bag_fn=outbag, docker_image_sha=docker_image_sha,
                     docker_image=docker_image, container_uuid="None",
                     mission_sha=mission_sha, mission_fn=mission_fn,
                     mutation_sha=delay_sha, mutation_fn=delay_fn,
                     context=str(sources), cursor=cursor, conn=conn)
    cmd = shlex.split(f'rosrunner --bag {outbag} --verbose --timeout {timeout} --topics "{topic_regex}" {param_fn}')
    subprocess.run(cmd)


def run_experiments(cursor, conn, param_fn: str,
                    docker_image=None, sources=None, num_iter=1):

    rsw = roswire.ROSWire()
    description = rsw.descriptions.load_or_build(docker_image, sources)
    docker_image_sha = description.sha256

    # Note: this assumes that the mission file is the first file copied
    # to the container
    mission_fn = get_from_yaml(param_fn, "files")[0]["host"]
    mission_sha = file_hash(mission_fn)

    for i in range(num_iter):
        bag_fn = f"{get_bag_fn()}.bag"

        run_one_experiment(sources, cursor, conn, outbag=bag_fn,
                           docker_image=docker_image,
                           docker_image_sha=docker_image_sha,
                           param_fn=param_fn, mission_sha=mission_sha,
                           mission_fn=mission_fn)


def get_from_yaml(fn: str, field: str) -> Any:
    with open(fn, 'r') as y:
        yaml_data = yaml.load(y, Loader=yaml.BaseLoader)
        field_data = yaml_data[field]
    return field_data


def main() -> None:
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=args.log_fn)
    ch = logging.StreamHandler()
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = "%m/%d/%Y %I:%M%S %p"
    fh_formatter = logging.Formatter(fmt=format_str, datefmt=date_str)
    fh.setFormatter(fh_formatter)
    ch.setFormatter(fh_formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Set up the bag database
    cursor, conn = access_bag_db(args.db_fn)

    for param_fn in args.rosrunner_yaml:
        # Warm up the docker image
        rsw = roswire.ROSWire()
        docker_image = get_from_yaml(param_fn, "image")
        sources = get_from_yaml(param_fn, "sources")
        logging.debug(f"Image: {docker_image}")
        logging.debug(f"Sources: {sources}")


        # Run the image with ROSRunner
        run_experiments(cursor, conn, param_fn, docker_image=docker_image,
                        sources=sources, num_iter=args.baseline_iterations)


if __name__ == '__main__':
    main()
