# -*- coding: utf-8 -*-
import time
import logging

import roswire

NAME_IMAGE = 'dskatz/mavros:example'
DIR_ROS_WS = '/ros_ws'


def main():
    # setup logging
    log_to_stdout = logging.StreamHandler()
    log_to_stdout.setLevel(logging.DEBUG)
    logging.getLogger('roswire').addHandler(log_to_stdout)

    rsw = roswire.ROSWire()

    # builds a description of the ROS application
    desc = rsw.descriptions.load(NAME_IMAGE)
    db_type = desc.types
    db_fmt = desc.formats
    print(desc)

    # launch a container for the ROS application
    with rsw.launch(NAME_IMAGE) as sut:
        # do some fun things with the file system
        # sut.files.read('/foo/bar')
        # sut.files.write('/foo/bar', 'blah')

        # optionally, build via catkin
        catkin = sut.catkin(DIR_ROS_WS)
        # catkin.clean()
        catkin.build()

        # launch the ROS master inside the container
        with sut.roscore() as roscore:

            # start ArduPilot SITL
            cmd = '/ros_ws/src/ArduPilot/build/sitl/bin/arducopter --model copter'
            sut.shell.non_blocking_execute(cmd)

            # launch MAVROS and connect to SITL
            roscore.launch('/ros_ws/src/mavros/mavros/launch/apm.launch',
                           args={'fcu_url': 'tcp://127.0.0.1:5760@5760'})
            time.sleep(10)

            # check that /mavros exists
            assert '/mavros' in roscore.nodes

            # import message type from mavros
            SetModeRequest = db_type['mavros_msgs/SetModeRequest']
            message = SetModeRequest(base_mode=64, custom_mode='')
            response = roscore.services['/mavros/set_mode'].call(message)
            assert response.response

            # let's record to a rosbag!
            with roscore.record('some_local_path') as recorder:
                time.sleep(10)

            # with roscore.playback() as player:
            #     pass


if __name__ == '__main__':
    main()
