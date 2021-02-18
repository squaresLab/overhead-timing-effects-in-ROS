import argparse
import copy
import logging
import os
import random
from typing import Any, Dict, List
import uuid
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--num_waypoints', type=int, default=4)
    parser.add_argument('--param_fn', type=str)
    parser.add_argument('--out_base', type=str, default="TEST")
    parser.add_argument('--waypoint_dir', type=str, default="waypoints")
    parser.add_argument('--same_end', action="store_false", default=True)
    args = parser.parse_args()
    return args


def parse_pgm(fn: str) -> List[List[int]]:
    pgm_matrix: List[List[int]] = []
    width = -1
    height = -1
    maxval = -1
    with open(fn, 'rb') as pgm_file:
        for line in pgm_file:
            if line.startswith(b'#'):
                continue
            if line.startswith(b'P5'):
                print(f"Magic number P5: {line}")
                continue
            if width == -1 and height == -1:
                split = line.split()
                width = int(split[0])
                height = int(split[1])
                print(f"Width: {width} and height {height}, line: {line}")
                continue
            if maxval == -1:
                maxval = int(line)
                print(f"Maxval: {maxval}")
            else:
                content = bytearray(line)
                length = len(content)
                print(f"length: {length}")
                assert(length/width == height)
                assert(length/height == width)
        end = width
        tmp_content = content
        while end <= length:
            sub_content = tmp_content[0:width]
            tmp_content = tmp_content[width:]
            end += width
            # print(f"sub_content: {sub_content}")
            # 1 represents white a.k.a. available pixels
            sub_list = [1 if x == maxval - 1 else 0 for x in sub_content]
            assert(len(sub_list) == width)
            pgm_matrix.append(sub_list)
        assert(len(pgm_matrix) == height), f"len(pgm_matrix): {len(pgm_matrix)} height: {height}"
    # print(pgm_matrix)
    assert(len(pgm_matrix) == height)
    assert(len(pgm_matrix[0]) == width)
    pgm_from_matrix(pgm_matrix, "TEST.pgm", width, height, 2)
    return pgm_matrix


def matrix_to_binary(pgm_matrix: List[List[int]]) -> bytearray:
    content = bytes()
    for row in pgm_matrix:
        row_bytes = bytearray(row)
        content += row_bytes
    return bytearray(content)


def pgm_from_matrix(pgm_matrix: List[List[int]], fn: str,
                    width: int, height: int, maxval: int) -> None:
    with open(fn, 'w') as pgm_file:
        pgm_file.write("P5\n")
        pgm_file.write(f"{width} {height}\n")
        pgm_file.write(f"{maxval}\n")
    with open(fn, 'ab') as pgm_bfile:
        binary_matrix = matrix_to_binary(pgm_matrix)
        pgm_bfile.write(bytes(binary_matrix))


def copy_matrix(in_matrix: List[List[int]]) -> List[List[int]]:
    out_matrix = []
    for row in in_matrix:
        new_row = []
        for item in row:
            new_row.append(item)
        out_matrix.append(new_row)
    return out_matrix


def get_header(seq:int =1, secs:int =1587484296, nsecs:int =938051083,
               frame_id:str ="map") -> Dict[str, Any]:
    header = dict()
    header["seq"] = seq
    times = dict()
    times["secs"] = secs
    times["nsecs"] = nsecs
    header["stamp"] = times
    header["frame_id"] = frame_id
    return header


def yaml_from_waypoint(pose, fn, seq=1):
    to_yaml = dict()
    to_yaml["header"] = get_header(seq=seq)
    to_yaml["pose"] = pose
    yaml_from_dict(to_yaml, fn)


def yaml_from_dict(to_yaml, fn):
    with open(fn, 'w') as yaml_file:
        yaml.dump(to_yaml, yaml_file)


def get_waypoint(bw_matrix: List[List[int]], resolution: float,
                 origin_x: float,
                 origin_y: float,
                 base_fn: str="TEST") -> Dict[str, Dict[str, float]]:
    height = len(bw_matrix)
    width = len(bw_matrix[0])

    while(True):
        y = random.randrange(height)
        x = random.randrange(width)
        z = 0
        position = {"x": (x * resolution) + origin_x,
                    "y": origin_y + ((height - y) * resolution),
                    "z": z}
        assert(waypoint_in_cosmap(position["x"], position["y"],
                                  width, height,
                                  origin_x, origin_y,
                                  resolution)), position
        print(f"\n\nposition: {position}")
        print(f"x: {x} y: {y}")
        #if bw_matrix[x][y] == 1:
        neighbor_occupied = False
        for i in range(-5, 5):
            if (y+i) < 0 or (y+i) >= len(bw_matrix):
                neighbor_occupied = True
                continue
            for j in range(-5, 5):
                if (x+j) < 0 or (x+j) >= len(bw_matrix[i]):
                    neighbor_occupied = True
                    continue
                if bw_matrix[y+i][x+j] == 0:
                    neighbor_occupied = True
        if not neighbor_occupied:
            break
        print(f"picking new waypoint because {x},{y} or neighbor is occupied")

    # Print a file with the waypoint location
    waypoint_matrix = copy_matrix(bw_matrix)
    #waypoint_matrix[x][y] = 0
    waypoint_matrix[y][x] = 0
    for i in range(-5, 5):
        waypoint_matrix[y+i][x] = 0
        waypoint_matrix[y][x+i] = 0
    pgm_from_matrix(waypoint_matrix, f"{base_fn}_{x}_{y}.pgm", width, height, 2)

    #position = {"x": (x * resolution) + origin_x,
    #            "y": origin_y + (height * resolution) - (y * resolution),
    #            "z": z}
    orientation = {"x": 0.0, "y": 0.0, "z":  0.688267569615, "w": 0.725456926782}
    pose = {"position": position, "orientation": orientation}

    yaml_from_waypoint(pose, f"{base_fn}_{x}_{y}.yml")

    return pose


def waypoint_in_cosmap(x: float, y: float,
                       pgm_width: int, pgm_height: int,
                       origin_x: float, origin_y: float,
                       resolution: float) -> bool:

    min_x = origin_x
    min_y = origin_y
    max_x = origin_x + (pgm_width * resolution)
    max_y = origin_y + (pgm_height * resolution)

    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False
    else:
        return True


def waypoints_to_pose_array(waypoints: List[Dict[str, Dict[str, float]]],
                            header) -> Dict[str, Any]:
    print(f"waypoints[0]: {waypoints[0]}")
    poses = [x for x in waypoints]
    pose_array = dict()
    pose_array["header"] = header
    pose_array["poses"] = poses
    return pose_array


def read_yaml(param_fn):
    with open(param_fn, 'r') as y:
        data = yaml.load(y, Loader=yaml.BaseLoader)
    return data


def main():
    args = parse_args()
    if not os.path.isdir(args.waypoint_dir):
        os.makedirs(args.waypoint_dir)

    bw_matrix = parse_pgm(args.in_file)
    waypoints = []

    parameters = read_yaml(args.param_fn)
    resolution = float(parameters["resolution"])
    origin_x = float(parameters["origin"][0])
    origin_y = float(parameters["origin"][1])

    base_fn = os.path.join(args.waypoint_dir, args.out_base)
    for i in range(args.num_waypoints):
        waypoint = get_waypoint(bw_matrix, resolution, origin_x, origin_y,
                                base_fn=base_fn)
        waypoints.append(waypoint)
    print(waypoints)

    # If we want the start and finish in the same place, copy the start
    # waypoint to the end
    if args.same_end:
        waypoints.append(copy.deepcopy(waypoints[0]))
    pose_array = waypoints_to_pose_array(waypoints, get_header())
    uuid_id = uuid.uuid4().hex
    pose_fn = f"{base_fn}_pose_array_{args.num_waypoints}_{uuid_id}.yaml"
    yaml_from_dict(pose_array, pose_fn)


if __name__ == '__main__':
    main()
