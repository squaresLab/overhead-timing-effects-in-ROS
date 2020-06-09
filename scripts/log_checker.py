import argparse
import logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_fn", type=str, action="append",
                        help="names of log files to check")
    parser.add_argument("--logging", type=str, default="checker.log",
                        help="filename for logging output from this script")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    log_stream = logging.StreamHandler()
    log_file = logging.FileHandler(args.logging)
    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(handlers=[log_stream, log_file], level=logging.DEBUG,
                        format=format_str, datefmt=date_str)

    logs = log_analysis.get_logs(args)

    for log_type in logs:
        pass

if __name__ == '__main__':
    main()
