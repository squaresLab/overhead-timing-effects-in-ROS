import argparse
import logging
import time

import roswire


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker_image", type=str, default="dskatz/husky_waypoints_ground_truth:new")
    parser.add_argument("--sources", type=str, action='append',
                        default=['/opt/ros/melodic/setup.bash'])
    parser.add_argument("--log_fn", type=str, default="remappings.log")
    args = parser.parse_args()
    return args


def main():
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

    rsw = roswire.ROSWire()
    description = rsw.descriptions.load_or_build(args.docker_image,
                                                 args.sources)

    print(type(rsw))

    with rsw.launch(args.docker_image, args.sources) as system:
        with system.roscore() as ros:
            launch_list = [("husky_gazebo", "playpen.launch"),
                           ("husky_gazebo", "spawn_husky.launch")]
#                           ("husky_navigation", "amcl_demo.launch")]

            print(f"type(ros): {type(ros)}")
            print(f"type(rsw): {type(rsw)}")

#            for package, fn in launch_list:
#              print(f"\n{package} {fn}")

                # launch_config = ros.roslaunch.read(package=package,
                #                                filename=fn)
                # print(launch_config)

            remappings = {"gazebo": [("/gazebo/model_states", "/huh"),
                                     ("/husky_velocity_controller/odom",
                                      "/_odom_orig")]}

            roslaunch_mgr = ros.roslaunch(filename="playpen.launch",
                                          package="husky_gazebo",
                                          node_to_remappings=remappings)

            roslaunch_mgr = ros.roslaunch(filename="spawn_husky.launch",
                                          package="husky_gazebo")
            time.sleep(40)
            system_state = ros.state
            print(system_state)


if __name__ == '__main__':
    main()
