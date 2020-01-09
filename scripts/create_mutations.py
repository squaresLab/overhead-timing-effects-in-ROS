import argparse
from comby import Comby
import difflib
import sys

# This script takes in a set of files and generates source code
# mutations according to the chosen strategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--returns', action="store_true", default=False,
                        help="Add instrumentation point before " +
                        "return statements")
    parser.add_argument('--files', action='append', type=str,
                        help="Specify the files to instrument")
    parser.add_argument('--delay', type=float, default=0.5,
                        help="Delay(secs) to insert at each insertion point.")
    parser.add_argument('--weight', type=float, default=0.5,
                        help="Portion of identified locations to instrument (between 0.0 and 1.0)")
    args = parser.parse_args()
    return args


def mutate_files(files, returns=False, delay=0.5):
    comby = Comby()
    if returns:
        match = "return :[x];"
        rewrite = "{sleep %f; return :[x];}" % delay
    else:
        raise NotImplemented
    for fn in files:
        with open(fn) as old:
            source_old = old.read()
            source_new = comby.rewrite(source_old, match, rewrite)
            fn_new = "%s_%f.new" % (fn, delay)
            with open(fn_new, 'w') as file_new:
                file_new.write(source_new)
        with open(fn) as old:
            fromlines = old.readlines()
        with open(fn_new) as new:
            tolines = new.readlines()

        diff = difflib.unified_diff(fromlines, tolines, fn, fn)

        diff_fn = "%s.diff" % fn
        with open(diff_fn, 'w') as diff_file:
            diff_file.writelines(diff)


def main():
    args = parse_args()

    if args.files:
        mutations = mutate_files(args.files, returns=args.returns)


if __name__ == '__main__':
    main()
