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
    parser.add_argument("--log_type", type=str, default="ardu")
    parser.add_argument("--bag_dir", type=str,
                        help="Use all files ending in .bag in the specified directory.")
    parser.add_argument("--nominal_delay", action="store_true",
                        default=False,
                        help="make a graph separating nominal runs from delayed")
    parser.add_argument("--alt_bag_base", type=str,
                        help="specify where the bag files referenced in the log_db reside")
    args = parser.parse_args()
    return args


def graph_logs_nominal_delay(one_mission, mission_fn="", mutation_fn="",
                             log_type="ardu"):
    print(one_mission)

    raise NotImplementedError


def graph_one_log(log: np.array, fn: str = "FIG.png", mutation_fn: str = "None",
                  mission_fn: str = "None", log_type: str = "ardu") -> None:
    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)

    if log_type == "ardu":
        lat, lon, time_elapsed, alt, relative_alt = extract_series(log)

        ax.scatter(lat, lon, c=time_elapsed, s=(relative_alt/100))
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Longitude")
    elif log_type == "husky":
        x, y, z, time_elapsed = extract_series(log, log_type=log_type)
        ax.scatter(x, y, c=time_elapsed, s=z)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    if mission_fn != "None":
            ax.set_title(mission_fn)
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


def extract_series(log, log_type="ardu", discard_zero: bool=True) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    if discard_zero:
        log = np.array([x for x in log if (x[1] != 0 and x[2] != 0)])
    else:
        log = np.array([x for x in log])

    if log_type == "ardu":
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

    elif log_type == "husky":
        x = log[:,1]
        y = log[:,2]
        z = log[:,3]
        time_elapsed = log[:,0]
        return (x, y, z, time_elapsed)


def get_delay_weight(mutation_fn: str,
                     log_type:str ="ardu") -> Tuple[float, float]:
    logging.debug(f"mutation_fn: {mutation_fn}")


    if log_type == "ardu":
        if mutation_fn.endswith(".diff"):
            mutation_fn = mutation_fn[:-5]

        split = mutation_fn.split("_")

        logging.debug(f"split: {split}")
        for section in split:
            logging.debug(f"section: {section}")
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

    elif log_type == "husky":
        weight = 1.0
        if "/" in mutation_fn:
            mutation_fn = mutation_fn.split("/")[-1]
        mutation_fn = mutation_fn.replace("husky_waypoint_ground_truth_remap_", "")
        logging.debug(f"mutation_fn (short): {mutation_fn}")
        fn_split = mutation_fn.split("_")
        try:
            delay = float(fn_split[-2])
        except ValueError:
            logging.debug(f"{fn_split[-2]} is not a float for a delay amount")

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
               use_external_time: bool=True, log_type="ardu") -> None:
    delays = []
    weights = []
    times = []
    for subset_label, logs_subset in logs.items():
        for log_fn, mutation_fn, log in logs_subset:
            if mutation_fn == "None" or mutation_fn == None:
                continue
            delay, weight = get_delay_weight(mutation_fn, log_type=log_type)
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
    plt.close(fig)


def graph_logs(logs: Dict[str, List[Tuple[str, str, np.array]]],
               log_type="ardu",
               mission_fn: str="",
               mutation_fn: str="") -> None:

    logging.debug(f"logs labels: {logs.keys()}")

    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)

    colors = [matplotlib.colors.to_rgb(x) for x in ('r', 'g', 'b', 'c', 'm', 'y')]
    color_maps = ["copper", "cool", "hot"] * 10

    zipped = zip(logs.keys(), colors, color_maps)
    title_short = mission_fn.split("/")[-1].split(".")[0]
    ax.set_title(f"{title_short} {logs.keys()}")
    legend_dict = dict()
    for subset_label, color, color_map in zipped:

        logging.debug(f"subset_label: {subset_label}")

        logs_subset = logs[subset_label]
        # Define the label
        # TODO
        #logging.debug(f"logs_subset[0]: {logs_subset[0]}")
        #logging.debug(f"type(logs_subset[0]): {type(logs_subset[0])}")

        y_min = 0
        x_min = 0
        y_max = 0
        x_max = 0

        for log_fn, mutation_fn, log in logs_subset:
            if log_type == "ardu":
                lat, lon, time_elapsed, \
                    alt, relative_alt = extract_series(log,
                                                       log_type=log_type)
                logging.debug(f"lat: {lat}")
                logging.debug(f"lon: {lon}")
                logging.debug(f"title_short: {title_short}")
                scatter = ax.scatter(lat, lon, c=time_elapsed,
                                     s=(relative_alt/100),
                           cmap=color_map)
                # ax.scatter(lat, lon, c=[[color] * len(lat)],
                #           s=(relative_alt/100))
                # logging.debug(f"color: {color}")
                # logging.debug(f"type(color): {type(color)}")
                if min(lon) < y_min or y_min == 0:
                    y_min = min(lon)
                if min(lat) < x_min or x_min == 0:
                    x_min = min(lat)
                if max(lon) > y_max or y_max == 0:
                    y_max = max(lon)
                if max(lat) > x_max or x_max == 0:
                    x_max = max(lat)
                ax.set_xlim(min(lat), max(lat))
                ax.set_xlabel("Latitude")
                ax.set_ylabel("Longitude")
                color = next_color(color)
            elif log_type == "husky":
                x, y, z, time_elapsed = extract_series(log, log_type=log_type)
                #logging.debug(f"x: {x}")
                #logging.debug(f"y: {y}")
                #logging.debug(f"title_short: {title_short}")
                # ax.scatter(x, y, c=time_elapsed, s=z)
                #color_array = [color] * len(x)
                #assert(len(color_array) == len(x)), f"{len(color_array)}, {len(x)}"
                legend_dict[subset_label] = ax.scatter(x, y,
                                                       c=time_elapsed,
                                                       cmap=color_map,
                                                       s=z)
                if min(y) < y_min or y_min == 0:
                    y_min = min(y)
                if min(x) < x_min or x_min == 0:
                    x_min = min(x)
                if max(y) > y_max or y_max == 0:
                    y_max = max(y)
                if max(x) > x_max or x_max == 0:
                    x_max = max(x)
                ax.set_xlabel("X position")
                ax.set_ylabel("Y position")
                color = next_color(color)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    for subset_label in legend_dict:
        cbar = fig.colorbar(legend_dict[subset_label])
        cbar.ax.set_xlabel(f"{subset_label}")
        cbar.ax.set_ylabel(f"total time elapsed")
    logging.debug(f"legend_dict: {legend_dict}")
    items = sorted(list(legend_dict.items()))
    logging.debug(f"items: {items}")
    fig.legend(handles=[x[1] for x in items],
               labels=[x[0] for x in items])

    fig.savefig(f"MANY_FIG_{title_short}.png")
    plt.close(fig)
    graph_time(logs, log_type=log_type)


def main() -> None:
    args = parse_args()

    log_stream = logging.StreamHandler()
    log_file = logging.FileHandler(args.logging)
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(handlers=[log_stream, log_file], level=logging.DEBUG,
                        format=format_str, datefmt=date_str)

    logs = log_analysis.get_logs(args)

    logs_by_mission = log_analysis.logs_by_mission(logs)

    # logging.debug(f"logs: {logs}")

    if (args.individual):
        for mission_fn, one_mission in logs_by_mission.items():
            logging.debug(f"mission filename: {mission_fn}")
            mission_fn_short = mission_fn.split("/")[-1].split(".")[0]
            filename_counter = 1
            for label, logs_subset in one_mission.items():
                for log_fn, mutation_fn, log in logs_subset:
                    log_fn_short = log_fn.split("/")[-1].split(".")[0]
                    filename = f"ONE_LOG_{log_fn_short}_mission_{mission_fn_short}.png"
                    mutation_fn_short = mutation_fn.split("/")[-1].split(".")[0]
                    graph_one_log(log, fn=filename,
                                  mutation_fn=mutation_fn_short,
                                  mission_fn=mission_fn_short,
                                  log_type=args.log_type)
                    filename_counter = filename_counter + 1

    for mission_fn, one_mission in logs_by_mission.items():
        mission_fn_short = mission_fn.split("/")[-1].split(".")[0]
        graph_logs(one_mission, mission_fn=mission_fn_short,
                   log_type=args.log_type)
        if args.nominal_delay:
            graph_logs_nominal_delay(one_mission, mission_fn=mission_fn_short,
                                     mutation_fn="None",
                                     log_type=args.log_type)

    #animate_logs(logs)


if __name__ == "__main__":
    main()
