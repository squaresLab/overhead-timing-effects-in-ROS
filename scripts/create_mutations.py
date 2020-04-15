import argparse
from comby import Comby
import difflib
import os
import random
import shlex
import shutil
import subprocess
import tempfile
import uuid

random.seed(42)

# This script takes in a set of files and generates source code
# mutations according to the chosen strategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--returns', action="store_true", default=False,
                        help="Add instrumentation point before " +
                        "return statements")
    parser.add_argument('--files', action='append', type=str,
                        help="Specify the files to instrument")
    parser.add_argument('--dir', type=str,
                        help="Instrument all cpp files in the directory.")
    parser.add_argument('--delay', type=float, default=0.5,
                        help="Delay(secs) to insert at each insertion point.")
    parser.add_argument('--weight', type=float, default=0.5,
                        help="Portion of identified locations to instrument (between 0.0 and 1.0)")
    parser.add_argument('--output', type=str, default="coin_flip")
    parser.add_argument('--sweep', action="store_true", default=False)
    parser.add_argument('--delay_min', type=int, default=-2,
                        help="minimum exponent in sweep delay (2^x)")
    parser.add_argument('--delay_max', type=int, default=4,
                        help="maximum exponent in sweep delay (2^x)")
    parser.add_argument('--publish', action="store_true", default=False,
                        help="Add instrumentation point before " +
                        "publish statements")
    args = parser.parse_args()
    return args


def get_fns(fn, delay, output="one_diff", weight=1.0, returns=False,
            publish=False):
    if returns:
        fn = f"{fn}_returns"
    if publish:
        fn = f"{fn}_publish"
    if output == "coin_flip":
        fn_new = f"{fn}_d{delay}_w{weight}.new"
        diff_fn = f"{fn}_d{delay}_w{weight}.diff"
    elif output == "one_diff":
        fn_new = "%s_all.new" % fn
        diff_fn = "%s_all.diff" % fn
    else:
        raise NotImplementedError

    return diff_fn, fn_new


def mutate_files(files, returns=False, publish=False, delay=0.5, weight=1.0,
                 output="coin_flip", old_dir=".", big_diff=False):
    old_dir = os.path.abspath(old_dir)
    comby = Comby()
    if returns:
        match = "return :[x];"
        rewrite = "{sleep %f; return :[x];}" % delay
    if publish:
        match = ":[y].publish(:[x])"
        rewrite = "{sleep %f; :[y].publish(:[x])}" % delay
    if not (returns or publish):
        raise NotImplementedError

    diff_fns = []
    new_fns = []
    #print(files)



    for fn in files:
        diff_fn, new_fn = get_fns(fn, delay, output=output, weight=weight,
                                  returns=returns, publish=publish)
        diff_fns.append(diff_fn)
        new_fns.append(new_fn)

        if os.path.isfile(new_fn) and os.path.isfile(diff_fn):
            continue
        with open(fn) as old:

            source_old = ""
            source_new = ""

            for line_old in old:
                if output=="one_diff" or (output=="coin_flip"
                                          and coin_flip(weight)):
                    line_new = comby.rewrite(line_old, match, rewrite,
                                             language=".cpp")
                    source_new += line_new
                else:
                    source_new += line_old
                source_old += line_old

        diff_fn, new_fn = create_output(fn, source_old, source_new, delay,
                                           new_fn,
                                           diff_fn,
                                           output=output,
                                           weight=weight)

    for fn in diff_fns + new_fns:
        cmd = f"dos2unix {fn}"
        subprocess.Popen(shlex.split(cmd))

    # Cat all the diff files together
    bdfn = f'dir_d{delay}_w{weight}.diff'
    big_diff_fn = os.path.join("patches", bdfn)
    cmd = "cat %s" % (" ".join(diff_fns))
    with open(big_diff_fn, 'w') as big_diff:
        print("running cmd: %s" % cmd)
        subprocess.Popen(shlex.split(cmd), stdout=big_diff)
        print("cat output to: %s" % big_diff_fn)


def coin_flip(weight):
    if random.random() < weight:
        return True
    else:
        return False


def create_output(fn, source_old, source_new, delay, fn_new, diff_fn,
                  output="one_diff",
                  weight=1.0):

    print("create_output: %s" % fn)

    with open(fn_new, 'w') as file_new:
        file_new.write(source_new)
    with open(fn) as old:
        fromlines = old.readlines()
    with open(fn_new) as new:
        tolines = new.readlines()

    print("calling unified_diff")
    diff = difflib.unified_diff(fromlines, tolines, os.path.basename(fn),
                                os.path.basename(fn), n=4)
    with open(diff_fn, 'w') as diff_file:
        print("writing to %s" % diff_fn)
        diff_file.writelines(diff)


    return diff_fn, fn_new


def main():
    args = parse_args()

    if args.files:
        files = args.files
    else:
        files = []

    if args.dir:
        contents = os.listdir(args.dir)
        cpp_files = [x for x in contents if x.endswith(".cpp")]
        files.extend([os.path.join(args.dir, x) for x in cpp_files])

    #print(files)

    if files:
        if args.sweep:
            weights = [x/10 for x in range(0, 10, 1)]
            #delays = [(1/pow(2, x)) for x in range(10)]

            #delays = [0.25, 0.5, 1, 2, 4, 8]
            delays = [pow(2, x) for x in range(args.delay_min, args.delay_max)]

            print(weights)
            print(delays)

            #print(files)
            for weight in weights:
                for delay in delays:
                    mutations = mutate_files(files, returns=args.returns,
                                             weight=weight, delay=delay,
                                             output=args.output,
                                             old_dir=args.dir,
                                             publish=args.publish)
        else:
            mutations = mutate_files(files, returns=args.returns,
                                     weight=args.weight, delay=args.delay,
                                     output=args.output, old_dir=args.dir)

if __name__ == '__main__':
    main()
