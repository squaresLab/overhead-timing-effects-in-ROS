import argparse
import logging

import roswire


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker_image", type=str, default="cob4")
    parser.add_argument("--log_fn", type=str, default="definitions.log")
    args = parser.parse_args()

    format_str = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    date_str = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(filename=args.log_fn, level=logging.DEBUG,
                        format=format_str, datefmt=date_str)

    rsw = roswire.ROSWire()
    description = rsw.descriptions.load_or_build(args.docker_image)
    for package in description.packages:
        logging.debug(package)

if __name__ == '__main__':
    main()
