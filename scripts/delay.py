#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

# Modified by Deby Katz 2020 for delay forwarding

import argparse
import rospy
import threading

from gazebo_msgs.msg import LinkStates
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import NavSatFix



# Hardcoded dictionary of topic name to msg type
# (The topic name corresponds to the *delayed_topic* because that's
# the original name of the topic in the system)

msg_type_dict = {"/gazebo/link_states": LinkStates,
                 "/_gazebo/link_states": LinkStates,
                 "/clock": Clock,
                 "/_clock": Clock,
                 "/husky_velocity_controller/odom": Odometry,
                 "/_husky_velocity_controller/odom": Odometry,
                 "/husky_velocity_controller/cmd_vel": Twist,
                 "/_husky_velocity_controller/cmd_vel": Twist,
                 "/navsat/fix": NavSatFix,
                 "_/navsat/fix": NavSatFix}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_topic", type=str, default="chatter")
    parser.add_argument("--delayed_topic", type=str, default="_chatter_delay")
    parser.add_argument("--delay_amount", type=int, default=2)
    parser.add_argument("--delay_probability", type=float, default=1.0)
    parser.add_argument("--queue_size", type=int, default=100)
    args = parser.parse_args()
    return args


def delayer(data, pub, delay_amount):
    rospy.sleep(delay_amount)
    rospy.loginfo(data)
    pub.publish(data)


def callback(data, callback_args):
    pub, delay_amount = callback_args
    p = threading.Thread(target=delayer, args=(data, pub, delay_amount))
    p.start()
    #p.join()


def talker(args):
    rospy.init_node('delay', anonymous=True)
    topic = msg_type_dict[args.delayed_topic]
    pub = rospy.Publisher(args.delayed_topic, topic,
                          queue_size=args.queue_size)
    sub = rospy.Subscriber(args.orig_topic, topic, callback,
                           callback_args=[pub, args.delay_amount],
                           queue_size=args.queue_size)

    rospy.spin()


if __name__ == '__main__':
    args = parse_args()
    try:
        talker(args)
    except rospy.ROSInterruptException:
        pass
