import argparse
import logging
import os
import sqlite3
from typing import List, Dict, Any, Optional

import roswire
from roswire.definitions import FormatDatabase, TypeDatabase

DIR_TEST = \
    '/usr0/home/dskatz/Documents/overhead-timing-effects-in-ROS/roswire/test/'


def access_bag_db(db_fn: str) -> sqlite3.Cursor:
    pass


def load_mavros_type_db():
    fn_db_format = os.path.join(DIR_TEST,
                                'format-databases/mavros.formats.yml')
    db_format = FormatDatabase.load(fn_db_format)
    return TypeDatabase.build(db_format)


def get_docker_image(args: argparse.Namespace) -> str:
    return args.docker_image


def get_mutations(args: argparse.Namespace) -> List[str]:
    patch_fns = args.patches
    context = args.context
    diffs = []
    for patch_fn in patch_fns:
        with open(patch_fn) as f:
            diff = f.read()
            diffs.append(diff)
    return diffs


def get_commands(wps: List[str]) -> List[Dict[str, str]]:
    commands = []
    for line in wps:
        if line.startswith("QGC WPL"):
            continue
        data = line.strip().split()
        labels = ["INDEX", "CURRENT_WP", "COORD_FRAME", "COMMAND", "PARAM1",
                  "PARAM2", "PARAM3", "PARAM4", "PARAM5", "PARAM6", "PARAM7",
                  "AUTOCONTINUE"]
        assert(len(data) == len(labels)), str("len(data): %d\ndata: %s" %
                                              (len(data), str(data)))
        labeled_command = zip(labels, data)
        one_command = dict(labeled_command)
        commands.append(one_command)
    return commands


def convert_waypoint(command: Dict[str, str]) -> Any:
    db_type = load_mavros_type_db()
    Waypoint = db_type['mavros_msgs/Waypoint']
    waypoint = Waypoint(frame=int(command["COORD_FRAME"]),
                        command=int(command["COMMAND"]),
                        is_current=False,
                        autocontinue=bool(command["AUTOCONTINUE"]),
                        param1=float(command["PARAM1"]),
                        param2=float(command["PARAM2"]),
                        param3=float(command["PARAM3"]),
                        param4=float(command["PARAM4"]),
                        x_lat=float(command["PARAM5"]),
                        y_long=float(command["PARAM6"]),
                        z_alt=float(command["PARAM7"]))
    return waypoint


def convert_mission(mission_fn: str) -> List[Any]:
    with open(mission_fn, 'r') as mission_file:
        wps = [x.strip() for x in mission_file.readlines()]
    commands = get_commands(wps)
    waypoints = []
    for command in commands:
        waypoint = convert_waypoint(command)
        waypoints.append(waypoint)
    return waypoints


def get_missions(mission_fns: List[str]) -> List[List[Any]]:
    missions = []
    for mission_fn in mission_fns:
        waypoints = convert_mission(mission_fn)
        missions.append(waypoints)
    return missions


def build_patched_system(system, diff: str, context: str):
    logging.debug("applying patch")
    system.files.patch(context, diff)
    logging.debug("patch applied")

    dir_workspace = '/ros_ws'

    catkin = system.catkin(dir_workspace)
    catkin.build()
    logging.debug("rebuilt")


def run_commands(system, mission: List[Any]) -> None:
    pass


def run_experiments(rsw, docker_image: str,
                    mutations: List[str], missions: List[List[Any]],
                    mutate: bool, context: str, baseline_iterations: int,
                    cursor) -> None:
    for mission in missions:
        with rsw.launch(docker_image) as system:
            for i in range(baseline_iterations):
                run_commands(system, mission)
            if mutate:
                for diff in mutations:
                    system = build_patched_system(system, diff, context)
                    run_commands(system, mission)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=str,
                        default='/ros_ws/src/ArduPilot/')
    parser.add_argument('--docker_image', type=str,
                        default='roswire/example:mavros')
    parser.add_argument('--patches', action='append', type=str,
                        help='Specify the patches to apply',
                        default=['patches/system.cpp.2.patch'])
    parser.add_argument('--log_fn', type=str, default='experiment.log')
    parser.add_argument('--mission_files', action='append', type=str,
                        help='Specify the mission files to convert',
                        default=['/usr0/home/dskatz/Documents/dsk-ardu-experimental-tools/good_missions/ba164dab.wpl'])
    parser.add_argument('--mutate', action='store_true', default=False)
    parser.add_argument('--db_fn', type=str, default='bag_db.db')
    parser.add_argument('--baseline_iterations', type=int, default=1)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(filename=args.log_fn)

    rsw = roswire.ROSWire()
    docker_image = get_docker_image(args)
    mutations = get_mutations(args)
    missions = get_missions(args.mission_files)

    cursor = access_bag_db(args.db_file)

    run_experiments(rsw, docker_image, mutations, missions, args.mutate,
                    args.context, args.baseline_iterations, cursor)


if __name__ == '__main__':
    main()
