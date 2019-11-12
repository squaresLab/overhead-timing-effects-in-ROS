import argparse
import logging
import os
import sqlite3
import time
from typing import List, Dict, Any, Optional
import uuid

import roswire
from roswire.definitions import FormatDatabase, TypeDatabase

DIR_TEST = \
    '/usr0/home/dskatz/Documents/overhead-timing-effects-in-ROS/roswire/test/'
FN_SITL = '/ros_ws/src/ArduPilot/build/sitl/bin/arducopter'
FN_PARAMS = '/ros_ws/src/ArduPilot/copter.parm'

bag_dir = '/usr0/home/dskatz/Documents/overhead-timing-effects-in-ROS/bags/'

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
    logging.info("applying patch")
    system.files.patch(context, diff)
    logging.info("patch applied")

    dir_workspace = '/ros_ws'

    catkin = system.catkin(dir_workspace)
    catkin.build()
    logging.info("rebuilt")
    return system


def run_commands(system, mission: List[Any], bag_fn: str) -> None:
    # Fetch dynamically generated types for the messages that we want to send
    SetModeRequest = system.messages['mavros_msgs/SetModeRequest']
    CommandBoolRequest = system.messages['mavros_msgs/CommandBoolRequest']
    CommandTOLRequest = system.messages['mavros_msgs/CommandTOLRequest']
    WaypointPushRequest = system.messages['mavros_msgs/WaypointPushRequest']

    # launch a temporary ROS session inside the app container
    # once the context is closed, the ROS session will be terminated and all
    # of its associated nodes will be automatically killed.
    logging.info("Running roscore")
    with system.roscore() as ros:
        # wait a bit for roscore
        time.sleep(10)

        # separately launch a software-in-the-loop simulator
        logging.info("Opening sitl")
        sitl_cmd = ("%s --model copter --defaults %s" % (FN_SITL, FN_PARAMS))
        ps_sitl = system.shell.popen(sitl_cmd)

        # use roslaunch to launch the application inside the ROS session
        ros.launch('apm.launch', package='mavros',
                   args={'fcu_url': 'tcp://127.0.0.1:5760@5760'})

        os.makedirs(bag_dir, exist_ok=True)
        with ros.record(os.path.join(bag_dir, bag_fn)) as recorder:
            # let's wait some time for the copter to become armable
            time.sleep(60)

            #request_manual = SetModeRequest(base_mode=0,
            #                              custom_mode='MANUAL')
            #response_manual = ros.services['/mavros/set_mode'].call(request_manual)
            #assert response_manual.success, str(response_manual)
            #logging.info("set_mode MANUAL successful")


            # arm the copter
            request_arm = CommandBoolRequest(value=True)
            response_arm = ros.services['/mavros/cmd/arming'].call(request_arm)
            assert response_arm.success
            logging.info("arm successful")

            # wait for the copter
            logging.info("waiting...")
            time.sleep(30)
            logging.info("finished waiting")

            request_auto = SetModeRequest(base_mode=0,
                                          custom_mode='AUTO')
            response_auto = ros.services['/mavros/set_mode'].call(request_auto)
            assert response_auto.success
            logging.info("set_mode AUTO successful")

            # Execute a mission
            logging.info(WaypointPushRequest.format.to_dict())
            logging.info("mission:\n%s\n" % str(mission))
            request_waypoint = WaypointPushRequest(waypoints=mission)
            logging.info("request_waypoint:\n%s\n\n" % str(request_waypoint))
            response_waypoint = ros.services['/mavros/mission/push'].call(request_waypoint)
            assert response_waypoint.success
            logging.info("WaypointPush successful")

            logging.info("waiting for copter to execute waypoints")
            logging.info("Waiting for copter to execute waypoints.")
            time.sleep(120)
            logging.info("Finished waiting for waypoints.")
            logging.info("finished waiting for waypoints")

        # Did we get to waypoints?

        # kill the simulator
        ps_sitl.kill()


def access_bag_db(db_fn: str) -> sqlite3.Cursor:

    # Check if there's the appropriate table. If not, make it
    sql_create_bagfns_table = """CREATE TABLE IF NOT EXISTS bagfns (
           bag_fn text PRIMARY KEY,
           image_sha text,
           image_name text,
           mission_sha text,
           context text
       ); """

    conn = sqlite3.connect(db_fn)

    if conn is not None:
        c = conn.cursor()
    else:
        raise("Error! Cannot create database connection!")

    try:
        c.execute(sql_create_bagfns_table)
    except sqlite3.Error as e:
        print(e)

    return c


def store_bag_fn(system, cursor, mission: str,
                 docker_image: str, context: str, bag_fn: str) -> None:
    docker_image_sha = system.description.sha256

    pass


def get_bag_fn() -> str:
    name = uuid.uuid4().hex
    while os.path.exists(os.path.join(bag_dir, name)):
        name = "%s.bag" % uuid.uuid4().hex
    return name


def execute_experiment(system, cursor, mission, docker_image, context):
    bag_fn = get_bag_fn()
    store_bag_fn(system, cursor, mission, docker_image, context, bag_fn)
    run_commands(system, mission, bag_fn)


def run_experiments(rsw, docker_image: str,
                    mutations: List[str], missions: List[List[Any]],
                    mutate: bool, context: str, baseline_iterations: int,
                    cursor) -> None:
    for mission in missions:
        with rsw.launch(docker_image) as system:
            for i in range(baseline_iterations):
                execute_experiment(system, cursor, mission, docker_image,
                                   context)
            if mutate:
                for diff in mutations:
                    system = build_patched_system(system, diff, context)
                    execute_experiment(system, cursor, mission, docker_image,
                                       context)


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
                        default=['missions/mair.mission.txt'])
    parser.add_argument('--mutate', action='store_true', default=False)
    parser.add_argument('--db_fn', type=str, default='bag_db.db')
    parser.add_argument('--baseline_iterations', type=int, default=1)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(filename=args.log_fn, level=logging.DEBUG,
                        format=format_str, datefmt=date_str)

    rsw = roswire.ROSWire()
    docker_image = get_docker_image(args)
    mutations = get_mutations(args)
    missions = get_missions(args.mission_files)

    cursor = access_bag_db(args.db_fn)

    run_experiments(rsw, docker_image, mutations, missions, args.mutate,
                    args.context, args.baseline_iterations, cursor)


if __name__ == '__main__':
    main()
