import argparse
import os

from roswire.definitions import (TypeDatabase, FormatDatabase,
                                 MsgFormat, Time, Duration)

# from test_bag import load_mavros_type_db

DIR_TEST = \
    '/usr0/home/dskatz/Documents/overhead-timing-effects-in-ROS/roswire/test/'


def load_mavros_type_db():
    fn_db_format = os.path.join(DIR_TEST,
                                'format-databases/mavros.formats.yml')
    db_format = FormatDatabase.load(fn_db_format)
    return TypeDatabase.build(db_format)


def test_encode_and_decode():
    db_type = load_mavros_type_db()
    Header = db_type['std_msgs/Header']
    orig = Header(seq=32, stamp=Time(secs=9781, nsecs=321), frame_id='')
    assert (orig == Header.decode(orig.encode()))


def get_commands(wps):
    commands = []
    for line in wps:
        if line.startswith("QGC WPL"):
            continue
        data = line.strip().split()
        labels = ["INDEX", "CURRENT_WP", "COORD_FRAME", "COMMAND", "PARAM1",
                  "PARAM2", "PARAM3", "PARAM4", "PARAM5", "PARAM6", "PARAM7",
                  "AUTOCONTINUE"]
        assert(len(data) == len(labels)), print("len(data): %d\ndata: %s" %
                                                (len(data), str(data)))
        labeled_command = zip(labels, data)
        one_command = dict(labeled_command)
        print(one_command)
        commands.append(one_command)
    return commands


def convert_waypoint(command, db_type):
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


def convert_mission(mission_fn):
    db_type = load_mavros_type_db()
    with open(mission_fn, 'r') as mission_file:
        wps = [x.strip() for x in mission_file.readlines()]
    commands = get_commands(wps)
    waypoints = []
    for command in commands:
        waypoint = convert_waypoint(command, db_type)
        waypoints.append(waypoint)
    return waypoints


def test_waypoint_encode_decode(mission_fn):
    db_type = load_mavros_type_db()
    with open(mission_fn, 'r') as mission_file:
        wps = mission_file.readlines()
    print(wps)
    commands = get_commands(wps)
    for command in commands:
        Waypoint = db_type['mavros_msgs/Waypoint']
        orig = Waypoint(frame=int(command["COORD_FRAME"]),
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
        print("\n\norig:\n%s" % str(orig))
        print("Waypoint.decode(orig.encode()):\n%s" %
              str(Waypoint.decode(orig.encode())))

        assert (orig == Waypoint.decode(orig.encode()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission_files', action='append', type=str,
                        help='Specify the mission files to convert',
                        default=['/usr0/home/dskatz/Documents/dsk-ardu-experimental-tools/good_missions/ba164dab.wpl'])
    args = parser.parse_args()
    return args


def main():
    # test_encode_and_decode()
    # mission_fn = \
    # '/usr0/home/dskatz/Documents/dsk-ardu-experimental-tools/good_missions/ba164dab.wpl'
    args = parse_args()
    for mission_fn in args.mission_files:
        waypoints = convert_mission(mission_fn)

    # test_waypoint_encode_decode(mission_fn)


if __name__ == '__main__':
    main()
