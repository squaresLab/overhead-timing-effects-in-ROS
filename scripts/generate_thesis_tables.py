import argparse
import logging
import sys
from typing import Dict, List, Tuple

import log_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str,
                        help="which graph to generate")
    args = parser.parse_args()
    return args


def waypoint_distance(args):
    # Get the set of experimental runs by mission
    # For each mission
    # For each delay amount in the mission?? (separate this way??)
    # For each execution in the mission
    # Calculate the minimum distance to each waypoint


    # Find the mean of the minimum distances for each waypoint
    # Find the standard deviation for the minimum distances for each WP
    # Find the final distances (maybe the same as the last waypoint)
    pass


def main():
    args = parse_args()

    if args.graph == "waypoint_distance":
        waypoint_distance(args)


if __name__ == '__main__':
    main()
