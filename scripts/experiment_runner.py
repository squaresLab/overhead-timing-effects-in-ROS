import argparse
import hashlib
from contextlib import closing, ExitStack

import logging
#from loguru import logger
import os
import sqlite3
import time
from typing import List, Dict, Any, Optional, Tuple
import uuid

import docker
import roswire
from roswire.definitions import FormatDatabase, TypeDatabase
#from roswire.proxy import ShellProxy as ROSWireShell
from dockerblade import Shell as ROSWireShell
from roswire.util import Stopwatch
from util import CircleIntBuffer


DIR_THIS = os.path.dirname(os.path.abspath(__file__))

FN_SITL = '/ros_ws/src/ArduPilot/build/sitl/bin/arducopter'
FN_PARAMS = '/ros_ws/src/ArduPilot/copter.parm'

bag_dir = '../bags/'


def build_shell(client_docker: docker.DockerClient,
                api_docker: docker.APIClient,
                uid_container: str
                ) -> ROSWireShell:
    container = client_docker.containers.get(uid_container)
    info = api_docker.inspect_container(uid_container)
    host_pid = int(info['State']['Pid'])
    return ROSWireShell(api_docker, container, host_pid)


def load_mavros_type_db():
    fn_db_format = os.path.join(DIR_THIS, '../test',
                                'mavros.formats.yml')
    db_format = FormatDatabase.load(fn_db_format)
    return TypeDatabase.build(db_format)


def get_docker_image(args: argparse.Namespace) -> str:
    return args.docker_image


def get_mutations(args: argparse.Namespace) -> List[Tuple[str, str]]:
    patch_fns = args.patches
    context = args.context
    diffs = []
    for patch_fn in patch_fns:
        with open(patch_fn) as f:
            diff = f.read()
            diffs.append((patch_fn, diff))
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


def build_patched_system(system, diff: str, context: str):
    logging.info("applying patch")
    system.files.patch(context, diff)
    logging.info("patch applied")

    dir_workspace = '/ros_ws'

    catkin = system.catkin(dir_workspace)
    catkin.build()
    logging.info("rebuilt")
    return system


def run_mavros(system, mission, ros):

    # use roslaunch to launch mavros inside the ROS session
    ros.roslaunch('apm.launch', package='mavros',
               args={'fcu_url': 'tcp://127.0.0.1:5760@5760'})

    # Fetch dynamically generated types for the messages that we want to send
    SetModeRequest = system.messages['mavros_msgs/SetModeRequest']
    CommandBoolRequest = system.messages['mavros_msgs/CommandBoolRequest']
    CommandTOLRequest = system.messages['mavros_msgs/CommandTOLRequest']
    CommandHomeRequest = system.messages['mavros_msgs/CommandHomeRequest']
    WaypointPushRequest = system.messages['mavros_msgs/WaypointPushRequest']
    CommandLongRequest = system.messages['mavros_msgs/CommandLongRequest']

    # let's wait some time for the copter to become armable
    time.sleep(60)

    # wait for the copter
    logging.info("waiting...")
    time.sleep(10)
    logging.info("finished waiting")

    # Execute a mission
    logging.info(WaypointPushRequest.format.to_dict())
    logging.info("mission:\n%s\n" % str(mission))
    request_waypoint = WaypointPushRequest(waypoints=mission)
    logging.info("request_waypoint:\n%s\n\n" % str(request_waypoint))
    wp_service = '/mavros/mission/push'
    response_waypoint = ros.services[wp_service].call(request_waypoint)
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
        0, 300, 0, 1, len(mission) + 1, 0, 0, 0, 0, 4)
    # 0, 0, 300, 0, 1, len(mission) + 1, 0, 0, 0, 0, 4)
    response_long = ros.services['/mavros/cmd/command'].call(request_long)
    assert response_long.success, response_long

    logging.info("waiting for copter to execute waypoints")
    logging.info("Waiting for copter to execute waypoints.")
    time.sleep(300)
    logging.info("Finished waiting for waypoints.")
    logging.info("finished waiting for waypoints")


def run_dronekit(system, mission_fn: str, mission_timeout=500):
    import ardu
    import dronekit
    
    with ExitStack() as exit_stack:

        ip = system.container.ip_address
        port = 5760
        url = "tcp:%s:%d" % (ip, port)
        print("url: %s" % url)

        vehicle = exit_stack.enter_context(
            closing(dronekit.connect(url, heartbeat_timeout=15)))
        wpl_mission = ardu.Mission.from_file(mission_fn)

        try:
            wpl_mission.execute(vehicle, timeout_mission=mission_timeout)
        except TimeoutError:
            logging.info("mission timed out")


def get_port_numbers(count, port_pool_mavlink):
    return port_pool_mavlink.take(3)


def mavproxy(system, mission_fn: str, logfile_name: str,
             port_pool_mavlink: CircleIntBuffer,
             exit_stack, timeout=1000) -> None:

    import ardu
    import dronekit
    
    # Code to work with mavproxy, adapted from trmo code
    uid_container = str(system.uuid)
    #api_client = exit_stack.enter_context(
    #    closing(docker.APIClient(base_url='unix://var/run/docker.sock')))
    #client_docker = exit_stack.enter_context(
    #    closing(docker.DockerClient(base_url='unix://var/run/docker.sock')))

    #shell = build_shell(client_docker, api_client, uid_container)
    shell = system.shell
    model = "copter"
    speedup = 1
    ports = get_port_numbers(3, port_pool_mavlink)
    logging.info("ports: %s" % str(ports))
    wpl_mission = ardu.Mission.from_file(mission_fn)
    sitl_kwargs = {'ip_address': system.container.ip_address,
                   'model': model,
                   'parameters_filename': FN_PARAMS,
                   'home': wpl_mission.home_location,
                   'speedup': speedup,
                   'ports': ports,
                   'logfile_name': logfile_name}
    url_dronekit, url_attacker, url_monitor = \
        exit_stack.enter_context(ardu.SITL.launch_with_mavproxy(shell,
                                                                **sitl_kwargs))

    logging.info("allocated DroneKit URL: %s", url_dronekit)
    logging.info("allocated attacker URL: %s", url_attacker)
    logging.info("allocated monitor URL: %s", url_monitor)

    # connect via DroneKit
    vehicle = exit_stack.enter_context(
        closing(dronekit.connect(url_dronekit, heartbeat_timeout=150)))

    # execute the mission
    timer = Stopwatch()
    timer.start()
    try:
        wpl_mission.execute(vehicle, timeout_mission=timeout)
    except TimeoutError:
        logging.info("mission timed out after %.2f seconds",
                      timer.duration)
        passed = False
    # allow a small amount of time for the message to arrive
    else:
        time.sleep(10)
    timer.stop()


def run_commands(system, mission_fn: str, bag_fn: str,
                 home: Dict[str, float], use_dronekit: bool,
                 use_mavproxy: bool,
                 port_pool_mavlink: CircleIntBuffer) -> None:

    # launch a temporary ROS session inside the app container
    # once the context is closed, the ROS session will be terminated and all
    # of its associated nodes will be automatically killed.
    logging.info("Running roscore")
    with system.roscore() as ros:

        with ExitStack() as exit_stack:

            # wait a bit for roscore
            time.sleep(10)

            bag_dir_abs = os.path.join(DIR_THIS, bag_dir)
            os.makedirs(bag_dir_abs, exist_ok=True)
            bag_fn_abs = os.path.join(bag_dir_abs, bag_fn)

            if use_mavproxy:
                mavproxy(system, mission_fn, bag_fn_abs, port_pool_mavlink,
                         exit_stack)

            else:
                # separately launch a software-in-the-loop simulator
                logging.info("Opening sitl")
                sitl_cmd = (("%s --model copter --home %f,%f,%f,%f" +
                            " --defaults %s") %
                            (FN_SITL, home['lat'], home['long'],
                             home['alt'], 270.0, FN_PARAMS))
                print(sitl_cmd)
                ps_sitl = system.shell.popen(sitl_cmd)

                if use_dronekit:
                    run_dronekit(system, mission_fn)
                else:
                    raise NotImplementedError("the mission filename")
                    with ros.record(bag_fn_abs) as recorder:
                        run_mavros(system, mission_fn, ros)

                # Did we get to waypoints?

                # kill the simulator
                ps_sitl.kill()


def access_bag_db(db_fn: str) -> Tuple[sqlite3.Cursor, sqlite3.Connection]:

    # Check if there's the appropriate table. If not, make it
    sql_create_bagfns_table = """CREATE TABLE IF NOT EXISTS bagfns (
           bag_fn text PRIMARY KEY,
           image_sha text,
           image_name text,
           container_uuid text,
           mission_sha text,
           mission_fn text,
           mutation_sha txt,
           mutation_fn txt,
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
        logging.warning(e)
    conn.commit()


    return c, conn

def file_hash(filename: str) -> str:
    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    with open(filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    sha = hasher.hexdigest()
    return sha


def store_bag_fn(system, cursor, conn, mission_fn: str,
                 docker_image: str, context: str, bag_fn: str,
                 mutation_fn: str) -> None:
    docker_image_sha = system.description.sha256
    container_uuid = str(system.uuid)
    print("container_uuid: %s" % container_uuid)

    mission_sha = file_hash(mission_fn)
    if mutation_fn is not "None":
        mutation_sha = file_hash(mutation_fn)
    else:
        mutation_sha = "None"
    command = "INSERT INTO bagfns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    values = (bag_fn, docker_image_sha, docker_image, container_uuid,
              mission_sha, mission_fn, mutation_sha, mutation_fn, context)
    cursor.execute(command, values)
    conn.commit()


def get_bag_fn() -> str:
    name = uuid.uuid4().hex
    while os.path.exists(os.path.join(bag_dir, name)):
        name = "%s.bag" % uuid.uuid4().hex
    return name


def execute_experiment(system, cursor, conn, mission_fn: str,
                       docker_image, context, home,
                       use_dronekit: bool, use_mavproxy: bool,
                       port_pool_mavlink: CircleIntBuffer,
                       mutation_fn="None") -> None:
    bag_fn = get_bag_fn()
    store_bag_fn(system=system, cursor=cursor, conn=conn,
                 mission_fn=mission_fn,
                 docker_image=docker_image, context=context, bag_fn=bag_fn,
                 mutation_fn=mutation_fn)
    run_commands(system, mission_fn, bag_fn, home, use_dronekit, use_mavproxy,
                 port_pool_mavlink)


def run_experiments(rsw, docker_image: str,
                    mutations: List[Any], mission_files: List[str],
                    mutate: bool, context: str, baseline_iterations: int,
                    cursor, conn, home, use_dronekit: bool,
                    use_mavproxy: bool) -> None:
    port_pool_mavlink = CircleIntBuffer(13000, 135000)

    sources = ['/opt/ros/indigo/setup.bash', '/ros_ws/devel/setup.bash']

    for mission_fn in mission_files:
        for i in range(baseline_iterations):
            logging.info(f"baseline iteration {i + 1} of {baseline_iterations}")
            print(f"baseline iteration {i + 1} of {baseline_iterations}")
            with rsw.launch(docker_image, sources) as system:
                execute_experiment(system, cursor, conn, mission_fn,
                                   docker_image,
                                   context, home, use_dronekit, use_mavproxy,
                                   port_pool_mavlink)

        if mutate:
            for diff_fn, diff in mutations:
                with rsw.launch(docker_image, sources) as system:
                    system = build_patched_system(system, diff, context)
                    execute_experiment(system, cursor, conn, mission_fn,
                                       docker_image, context, home,
                                       use_dronekit, use_mavproxy,
                                       port_pool_mavlink, mutation_fn=diff_fn)


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
    parser.add_argument('--home_lat', type=float,
                        default=42.2944644474907321)
    parser.add_argument('--home_long', type=float,
                        default=-83.7104686349630356)
    parser.add_argument('--home_alt', type=float, default=274.709991455078125)
    parser.add_argument('--use_dronekit', default=False, action='store_true')
    parser.add_argument('--use_mavproxy', default=False, action='store_true')
    args = parser.parse_args()

    if not args.patches:
        args.patches = [os.path.join(DIR_THIS, 'patches/system.cpp.2.patch')]
    if not args.mission_files:
        args.mission_files = [os.path.join(DIR_THIS,
                                           'missions/mair.mission.txt')]
    return args


def main() -> None:
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=args.log_fn)
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'

    fh_formatter = logging.Formatter(fmt=format_str, datefmt=date_str)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    #logging.basicConfig(filename=args.log_fn, level=logging.DEBUG,
    #                    format=format_str, datefmt=date_str)


    # TODO:
    # add boilerplate python default logging or use loguru to attach to the correct
    #logger.remove()
    #logger.enable('roswire')
    #logger.add(args.log_fn, level='INFO', format="{time:MM/DD/YYYY HH:mm:ss}:{level}:{name} {message}")

    rsw = roswire.ROSWire()
    docker_image = get_docker_image(args)
    mutations = get_mutations(args)
    missions = [convert_mission(fn) for fn in args.mission_files]

    cursor, conn = access_bag_db(args.db_fn)

    home = dict((('lat', args.home_lat), ('long', args.home_long),
                 ('alt', args.home_alt)))
    run_experiments(rsw, docker_image, mutations, args.mission_files,
                    args.mutate,
                    args.context, args.baseline_iterations, cursor, conn, home,
                    args.use_dronekit, args.use_mavproxy)
    conn.close()


if __name__ == '__main__':
    main()
