import argparse
from enum import Enum
import logging
from ruamel.yaml import YAML
import time

import roswire

FN_SITL = '/ros_ws/src/ArduPilot/build/sitl/bin/arducopter'
FN_PARAMS = '/ros_ws/src/ArduPilot/copter.parm'


def get_docker_image(args: argparse.Namespace) -> str:
    return "roswire/example:mavros"


def run_commands(system: roswire.system.System) -> None:
    # Fetch the dynamically generated types for the messages that we want to send
    SetModeRequest = system.messages['mavros_msgs/SetModeRequest']
    CommandBoolRequest = system.messages['mavros_msgs/CommandBoolRequest']
    CommandTOLRequest = system.messages['mavros_msgs/CommandTOLRequest']

    # launch a temporary ROS session inside the app container
    # once the context is closed, the ROS session will be terminated and all
    # of its associated nodes will be automatically killed.
    print("Running roscore")
    with system.roscore() as ros:
        # for this example, we need to separately launch a software-in-the-loop
        # simulator for the robot platform
        print("Opening sitl")
        ps_sitl = \
            system.shell.popen(f'{FN_SITL} --model copter --defaults {FN_PARAMS}')

        # use roslaunch to launch the application inside the ROS session
        ros.launch('apm.launch', package='mavros', args={'fcu_url': 'tcp://127.0.0.1:5760@5760'})

        # with ros.record('example_recorded_bag.bag') as recorder:
        # let's wait some time for the copter to become armable
        time.sleep(60)

        # arm the copter
        request_arm = CommandBoolRequest(value=True)
        response_arm = ros.services['/mavros/cmd/arming'].call(request_arm)
        assert response_arm.success
        print("arm successful")

        # switch to guided mode
        request_guided = SetModeRequest(base_mode=0, custom_mode='GUIDED')
        response_guided = ros.services['/mavros/set_mode'].call(request_guided)
        assert response_arm.success
        print("mode set success")

        # takeoff to 50 metres above the ground
        request_takeoff = CommandTOLRequest(min_pitch=0.0,
                                            yaw=0.0,
                                            latitude=0.0,
                                            longitude=0.0,
                                            altitude=50.0)
        response_takeoff = ros.services['/mavros/cmd/takeoff'].call(request_takeoff)
        assert response_takeoff.success

        # wait for the copter to reach the target altitude
        print("waiting for copter to reach altitude...")
        time.sleep(30)
        print("finished waiting")

        # Execute a mission??
        # DSK 11-4-19 -- you left off here

        # kill the simulator
        ps_sitl.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', action='append', type=str,
                        help='Specify the files to instrument')
    parser.add_argument('--patches', action='append', type=str,
                        help='Specify the patches to apply',
                        default=['system.cpp.2.patch'])
    parser.add_argument('--image', type=str,
                        default='/roswire/example:mavros')
    parser.add_argument('--context', type=str,
                        default='/ros_ws/src/ArduPilot/')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(filename="play_log.log")

    rsw = roswire.ROSWire()

    # Find the docker image
    docker_image = get_docker_image(args)

    patch_fns = args.patches
    context = '/ros_ws/src/ArduPilot'

    for patch_fn in patch_fns:
        with open(patch_fn) as f:
            diff = f.read()
        print("applying patch...")

        print("\n\nlaunching docker image: %s" % docker_image)

        # rebuild the docker container
        with rsw.launch(docker_image) as system:
            context = '/ros_ws/src/ArduPilot'

            print("printing diff")
            print(diff)

            print("printing context")
            print(context)

            print("applying patch")
            system.files.patch(context, diff)
            print("patch applied")

            print("rebuilding...")
            dir_workspace = '/ros_ws'

            catkin = system.catkin(dir_workspace)
            catkin.build()
            print("rebuilt")

            run_commands(system)


        # run tests via service calls (inputs, with and without instrumentation


    # extract logs via rosbag

    # analyze bag files


if __name__ == '__main__':
    main()
