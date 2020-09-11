import argparse
import copy
import os
import yaml

from experiment_runner import file_hash


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml", type=str, default="husky_waypoints_ground_truth_remap.yml")
    parser.add_argument("--delay_length", type=float, action='append',
                        default=[1.0])
    parser.add_argument("--delay_topic", type=str, action='append',
                        default=["/gazebo/link_states"])
    parser.add_argument("--mission_fn", type=str, default=[],
                        action='append')
    parser.add_argument("--mission_file_dir", type=str,
                        default="/usr0/home/dskatz/Documents/overhead-timing-effect-in-ROS/scripts/waypoint_husky/")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Read in the base YAML file
    with open(args.base_yaml, "r") as yaml_base_file:
        base_yaml = yaml.load(yaml_base_file, Loader=yaml.Loader)
    print(type(base_yaml))
    print(base_yaml)

    # for each topic to delay
    for topic in args.delay_topic:
        assert(topic.startswith("/")), topic
        topic_alt = topic.replace("/", "/_", 1)

        # for each delay length
        for delay_length in args.delay_length:

            for mission_fn in args.mission_fn:

                new_yaml = copy.deepcopy(base_yaml)
                print(type(new_yaml))
                index = range(len(new_yaml["instructions"]))
                for i, instruction in zip(index, new_yaml["instructions"]):
                    print(instruction)
                    if "type" in instruction and "shell" in instruction["type"] \
                            and "command" in instruction and \
                            "delay_topics delay.py" in instruction["command"]:
                        print("found relevant instruction")
                        delay_cmd = instruction["command"]
                        command_index = i
                        break
                    else:
                        print("not relevant instruction")
                new_delay_cmd = f"rosrun delay_topics delay.py --delayed_topic {topic} --orig_topic {topic_alt} --delay_amount {delay_length}"
                print(new_delay_cmd)
                new_yaml["instructions"][command_index]["command"] = \
                    new_delay_cmd
                mission_path = os.path.join(args.mission_file_dir, mission_fn)
                new_yaml["files"][0]["host"] = mission_path
                mission_sha = file_hash(mission_path)

                # write a new YAML file with the appropriate parameters
                base_fn, _, suffix = (args.base_yaml).rpartition(".")
                topic_no_slash = topic.replace("/", "-")
                new_fn = f"{base_fn}_{topic_no_slash}_{delay_length}_{mission_sha}.{suffix}"
                with open(new_fn, "w") as f:
                    yaml.dump(new_yaml, f)


if __name__ == '__main__':
    main()
