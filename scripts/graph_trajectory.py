import argparse
import datetime
import logging
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import ScalarFormatter
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

import log_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_db", type=str)
    parser.add_argument("--log_fn", type=str, action="append")
    parser.add_argument("--logging", type=str, default="graph.log")
    parser.add_argument("-i", "--individual", action="store_true",
                        default=False)
    args = parser.parse_args()
    return args


def graph_one_log(log: np.array, fn: str = "FIG.png", title: str = "None") -> None:
    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)

    lat, lon, time_elapsed, alt, relative_alt = extract_series(log)
    #lat = log[:,1][1:]
    #lon = log[:,2][1:]

    #logging.debug(f"lat: {lat}")
    #logging.debug(f"lon: {lon}")
    ax.scatter(lat, lon, c=time_elapsed, s=(relative_alt/100))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    if title != "None":
        ax.set_title(title)
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

    #logging.debug(f"next_color: {next_color}")
    return next_color


def animate_logs(logs) -> None:
    fig, ax = plt.subplots()
    line, = ax.scatter([], [], lw=2)
    plt.close(fig)


def extract_series(log, discard_zero: bool=True) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    if discard_zero:
        log = np.array([x for x in log if (x[1] != 0 and x[2] != 0)])
    else:
        log = np.array([x for x in log])

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


def get_delay_weight(mutation_fn: str) -> Tuple[float, float]:
    logging.debug(f"mutation_fn: {mutation_fn}")

    if mutation_fn.endswith(".diff"):
        mutation_fn = mutation_fn[:-5]

    split = mutation_fn.split("_")

    logging.debug(f"split: {split}")
    for section in split:
        logging.debug("section: {section}")
        if section.startswith("d") or section.startswith("w"):
            try:
                value = float(section[1:])
            except ValueError:
                logging.debug(f"{section[1:]} is not a float")
                continue
            logging.debug(f"value: {value}")
            if section.startswith("d"):
                delay = value
            elif section.startswith("w"):
                weight = value
            else:
                assert (False), "Incompatible if logic."

    return delay, weight


def get_total_time(log: np.array, use_external_time: bool=True) -> float:
    if not use_external_time:
        time_elapsed = log[:,0]
        total_time_ms = time_elapsed[-1]
        total_time = total_time_ms / 1000
    else:
        timestamps = log[:,-1]
        timestamp_initial = datetime.datetime.fromtimestamp(timestamps[1])
        timestamp_final = datetime.datetime.fromtimestamp(timestamps[-1])
        total_time = (timestamp_final - timestamp_initial).total_seconds()
    return total_time


def graph_time(logs: Dict[str, List[Tuple[str, str, np.array]]],
               use_external_time: bool=True) -> None:
    delays = []
    weights = []
    times = []
    for subset_label, logs_subset in logs.items():
        for log_fn, mutation_fn, log in logs_subset:
            if mutation_fn == "None" or mutation_fn == None:
                continue
            delay, weight = get_delay_weight(mutation_fn)
            delays.append(delay)
            weights.append(weight)
            total_time = get_total_time(log, use_external_time)
            times.append(total_time)

    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Delay amount (seconds, log scale)")
    ax.set_ylabel("Delay probability")
    ax.set_title("Total running time of ArduCopter")
    #ax.ticklabel_format(style="plain")
    scatter = ax.scatter(delays, weights, c=times, cmap="Oranges")
    cbar = fig.colorbar(scatter)
    cbar.ax.set_ylabel("total time elapsed (s)")
    fig.savefig("TOTAL_TIME.png")


def graph_logs(logs: Dict[str, List[Tuple[str, str, np.array]]]) -> None:

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
        #logging.debug(f"type(logs_subset[0]): {type(logs_subset[0])}")
        # Plot each log in a variation in the color family
        for log_fn, mutation_fn, log in logs_subset:
            lat, lon, time_elapsed, alt, relative_alt = extract_series(log)
            ax.scatter(lat, lon, c=time_elapsed, s=(relative_alt/100))
            # logging.debug(f"color: {color}")
            # logging.debug(f"type(color): {type(color)}")
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Longitude")
            color = next_color(color)

    fig.savefig("MANY_FIG.png")

    graph_time(logs)


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

    if (args.individual):
        for logs_subset in logs.values():
            filename_counter = 1
            for log_fn, mutation_fn, log in logs_subset:
                log_fn_short = log_fn.split("/")[-1].split(".")[0]
                filename = f"ONE_LOG_{log_fn_short}.png"
                graph_one_log(log, fn=filename, title=mutation_fn)
                filename_counter = filename_counter + 1

    graph_logs(logs)

    #animate_logs(logs)


if __name__ == "__main__":
    main()
