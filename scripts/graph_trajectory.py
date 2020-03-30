import argparse
import logging
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

import log_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_db", type=str)
    parser.add_argument("--log_fn", type=str, action="append")
    parser.add_argument("--logging", type=str, default="graph.log")
    args = parser.parse_args()
    return args


def graph_one_log(log: np.array, fn: str = "FIG.png") -> None:
    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)

    lat, lon, time_elapsed, alt, relative_alt = extract_series(log)
    #lat = log[:,1][1:]
    #lon = log[:,2][1:]

    logging.debug(f"lat: {lat}")
    logging.debug(f"lon: {lon}")
    ax.scatter(lat, lon, c=time_elapsed, s=(relative_alt/100))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    logging.debug(f"saving to filename: {fn}")
    fig.savefig(fn)
    plt.close(fig)


def next_color(in_color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    assert(in_color[0] <= 1 and in_color[0] >=0)
    if in_color[0] >= 0.5:
        next_color = (in_color[0] - 0.05, in_color[1], in_color[2])
    elif in_color[1] >= 0.5:
        next_color = (in_color[0], in_color[1] - 0.05, in_color[2])
    elif in_color[2] >= 0.5:
        next_color = (in_color[0], in_color[1], in_color[2] - 0.05)
    elif in_color[2] < 0.5:
        next_color = (in_color[0], in_color[1], in_color[2] + 0.05)
    elif in_color[1] < 0.5:
        next_color = (in_color[0], in_color[1] + 0.05, in_color[2])
    elif in_color[0] < 0.5:
        next_color = (in_color[0] + 0.05, in_color[1], in_color[2])
    else:
        next_color = in_color
    # TODO: the less than equivalents
    logging.debug(f"next_color: {next_color}")
    return next_color


def animate_logs(logs) -> None:
    fig, ax = plt.subplots()
    line, = ax.scatter([], [], lw=2)
    plt.close(fig)


def extract_series(log) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    log = np.array([x for x in log if (x[1] != 0 and x[2] != 0)])

    lat = log[:,1]
    lon = log[:,2]
    time_elapsed = log[:,0]
    alt = log[:,3]
    relative_alt = log[:,4]

    lat = [x/10000000 for x in lat]
    lon = [x/10000000 for x in lon]
    assert(all([x < 90 and x > -90 for x in lat])), lat
    assert(all([x < 180 and x > -180 for x in lon])), lon

    return (lat, lon, time_elapsed, alt, relative_alt)


def graph_logs(logs) -> None:

    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)
    colors = [matplotlib.colors.to_rgb(x) for x in ('r', 'g', 'b', 'c', 'm', 'y')]
    zipped = zip(logs.keys(), colors)
    for subset_label, color in zipped:
        # Pick a color family

        logs_subset = logs[subset_label]
        # Define the label
        # TODO
        #logging.debug(f"logs_subset[0]: {logs_subset[0]}")
        logging.debug(f"type(logs_subset[0]): {type(logs_subset[0])}")
        # Plot each log in a variation in the color family
        for log_fn, log in logs_subset:
            lat, lon, time_elapsed, alt, relative_alt = extract_series(log)
            ax.scatter(lat, lon, c=time_elapsed, s=(relative_alt/100))
            # logging.debug(f"color: {color}")
            # logging.debug(f"type(color): {type(color)}")
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Longitude")
            color = next_color(color)

    fig.savefig("MANY_FIG.png")


def main() -> None:
    args = parse_args()

    log_stream = logging.StreamHandler()
    log_file = logging.FileHandler(args.logging)
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(handlers=[log_stream, log_file], level=logging.DEBUG,
                        format=format_str, datefmt=date_str)

    logs = log_analysis.get_logs(args)

    # logging.debug(f"logs: {logs}")

    for logs_subset in logs.values():
        filename_counter = 1
        for log_fn, log in logs_subset:
            log_fn_short = log_fn.split("/")[-1].split(".")[0]
            filename = f"ONE_LOG_{log_fn_short}.png"
            graph_one_log(log, fn=filename)
            filename_counter = filename_counter + 1

    graph_logs(logs)

    #animate_logs(logs)


if __name__ == "__main__":
    main()
