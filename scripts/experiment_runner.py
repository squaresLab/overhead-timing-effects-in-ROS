import argparse
import logging
import os
import sqlite3
import time
from typing import List, Dict, Any, Optional
import uuid

import dronekit
import dronekit_sitl
import roswire
from roswire.definitions import FormatDatabase, TypeDatabase

import ardu

DIR_THIS = os.path.dirname(os.path.abspath(__file__))

FN_SITL = '/ros_ws/src/ArduPilot/build/sitl/bin/arducopter'
FN_PARAMS = '/ros_ws/src/ArduPilot/copter.parm'

bag_dir = '../bags/'

# This is hard-coded to match the mair mission. Change this.
home_tuple = (True, 42.2944644474907321, -83.7104686349630356,
              274.709991455078125)

def load_mavros_type_db():
    fn_db_format = os.path.join(DIR_THIS, '../test',
                                'mavros.formats.yml')
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


def run_roswire(system, mission: (str, List['Waypoint']), bag_fn: str) -> None:
    # Fetch dynamically generated types for the messages that we want to send
    SetModeRequest = system.messages['mavros_msgs/SetModeRequest']
    CommandBoolRequest = system.messages['mavros_msgs/CommandBoolRequest']
    CommandTOLRequest = system.messages['mavros_msgs/CommandTOLRequest']
    CommandHomeRequest = system.messages['mavros_msgs/CommandHomeRequest']
    WaypointPushRequest = system.messages['mavros_msgs/WaypointPushRequest']
    CommandLongRequest = system.messages['mavros_msgs/CommandLongRequest']
    # launch a temporary ROS session inside the app container
    # once the context is closed, the ROS session will be terminated and all
    # of its associated nodes will be automatically killed.
    logging.info("Running roscore")
    with system.roscore() as ros:
        # wait a bit for roscore
        time.sleep(10)

        # separately launch a software-in-the-loop simulator
        logging.info("Opening sitl")
        sitl_cmd = ("%s --model copter --defaults %s" %
                    (FN_SITL, FN_PARAMS))
        ps_sitl = system.shell.popen(sitl_cmd)

        # use roslaunch to launch the application inside the ROS session
        ros.launch('apm.launch', package='mavros',
                   args={'fcu_url': 'tcp://127.0.0.1:5760@5760'})

        bag_dr = os.path.join(DIR_THIS, bag_dir)
        os.makedirs(bag_dr, exist_ok=True)
        with ros.record(os.path.join(bag_dr, bag_fn)) as recorder:
            # let's wait some time for the copter to become armable
            time.sleep(60)

            #request_manual = SetModeRequest(base_mode=0,
            #                              custom_mode='MANUAL')
            #response_manual = ros.services['/mavros/set_mode'].call(request_manual)
            #assert response_manual.success, str(response_manual)
            #logging.info("set_mode MANUAL successful")

            request_home = CommandHomeRequest(*home_tuple)
            response_home = ros.services['/mavros/cmd/set_home'].call(request_home)
            assert response_home.success
            logging.info("successfully set home")

            # wait for the copter
            logging.info("waiting...")
            time.sleep(10)
            logging.info("finished waiting")

            # Execute a mission
            logging.info(WaypointPushRequest.format.to_dict())
            logging.info("mission:\n%s\n" % str(mission[1]))
            request_waypoint = WaypointPushRequest(waypoints=mission[1])
            logging.info("request_waypoint:\n%s\n\n" % str(request_waypoint))
            response_waypoint = ros.services['/mavros/mission/push'].call(request_waypoint)
            assert response_waypoint.success
            logging.info("WaypointPush successful")

            # wait for the copter
            logging.info("waiting...")
            time.sleep(10)
            logging.info("finished waiting")

            # arm the copter
            request_arm = CommandBoolRequest(value=True)
            response_arm = ros.services['/mavros/cmd/arming'].call(request_arm)
            assert response_arm.success
            logging.info("arm successful")

            # wait for the copter
            logging.info("waiting...")
            time.sleep(2)
            logging.info("finished waiting")
            request_auto = SetModeRequest(base_mode=0,
                                          custom_mode='AUTO')
            response_auto = ros.services['/mavros/set_mode'].call(request_auto)
            assert response_auto.success
            logging.info("set_mode AUTO successful")

            # wait for the copter
            logging.info("waiting...")
            time.sleep(2)
            logging.info("finished waiting")

            request_long = CommandLongRequest(
            0, 300, 0, 1, len(mission[1]) + 1, 0, 0, 0, 0, 4)
            # 0, 0, 300, 0, 1, len(mission) + 1, 0, 0, 0, 0, 4)
            response_long = ros.services['/mavros/cmd/command'].call(request_long)
            #assert response_long.success, response_long

            logging.info("waiting for copter to execute waypoints")
            logging.info("Waiting for copter to execute waypoints.")
            time.sleep(120)
            logging.info("Finished waiting for waypoints.")
            logging.info("finished waiting for waypoints")

        # Did we get to waypoints?

        # kill the simulator
        ps_sitl.kill()


def run_dronekit(mission):
    # launch SITL
    #sitl_kwargs = {'ip_address': ip_address,
    #               'model': model,
    #               'parameters_filename': parameters_filename,
    #               'home': mission.home_location,
    #               'speedup': speedup,
    #               'ports': ports_mavlink}
    #url_dronekit, url_attacker, url_monitor = \
    #    exit_stack.enter_context(SITL.launch_with_mavproxy(shell, **sitl_kwargs))
    logging.info("Opening sitl - dronekit")
    copter_args = ['-S', '--model', 'copter', '--home=42.2944644474907321,-83.7104686349630356,274.709991455078125,0']
    #sitl = dronekit_sitl.SITL()
    #sitl.download('copter', '3.3')
    #sitl.launch(copter_args)
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()
    print("Connecting to vehicle on: %s" % (connection_string))
    vehicle = dronekit.connect(connection_string, heartbeat_timeout=100,
                               wait_ready=True)
    print("Connected")
    mission_fn = mission[0]
    wpl_mission = ardu.Mission.from_file(mission_fn)

    try:
        wpl_mission.execute(vehicle, timeout_mission=500)
    except TimeoutError:
        logger.debug("mission timed out")

    vehicle.close()

    # Shut down simulator
    sitl.stop()
    print("Completed")


def run_commands(system, mission: List[Any], bag_fn: str,
                 use_roswire: bool) -> None:
    if use_roswire:
        run_roswire(system, mission, bag_fn)
    else:
        run_dronekit(mission)


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


def execute_experiment(system, cursor, mission,
                       docker_image, context,
                       use_roswire):
    bag_fn = get_bag_fn()
    store_bag_fn(system, cursor, mission, docker_image, context, bag_fn)
    run_commands(system, mission, bag_fn, use_roswire)


def run_experiments(rsw, docker_image: str,
                    mutations: List[str], missions: Dict[str, List[Any]],
                    mutate: bool, context: str, baseline_iterations: int,
                    cursor, use_roswire: bool) -> None:
    for mission in missions.items():
        with rsw.launch(docker_image) as system:
            for i in range(baseline_iterations):
                execute_experiment(system, cursor, mission, docker_image,
                                   context, use_roswire)
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
                        help='Specify the patches to apply')
    parser.add_argument('--log_fn', type=str, default='experiment.log')
    parser.add_argument('--mission_files', action='append', type=str,
                        help='Specify the mission files to convert')
    parser.add_argument('--mutate', action='store_true', default=False)
    parser.add_argument('--db_fn', type=str, default='bag_db.db')
    parser.add_argument('--baseline_iterations', type=int, default=1)
    parser.add_argument('--use_dronekit', default=False, action='store_true')
    args = parser.parse_args()

    if not args.patches:
        args.patches = [os.path.join(DIR_THIS, 'patches/system.cpp.2.patch')]
    if not args.mission_files:
        args.mission_files = [os.path.join(DIR_THIS,
                                           'missions/mair.mission.txt')]
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
    missions_dict = dict(zip(args.mission_files, missions))

    cursor = access_bag_db(args.db_fn)

    run_experiments(rsw, docker_image,
                    mutations, missions_dict,
                    args.mutate, args.context, args.baseline_iterations,
                    cursor, not args.use_dronekit)


if __name__ == '__main__':
    main()
