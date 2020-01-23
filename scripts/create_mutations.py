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
    args = parser.parse_args()
    return args


def get_fns(fn, delay, output="one_diff", weight=1.0):
    if output == "coin_flip":
        fn_new = f"{fn}_d{delay}_w{weight}.new"
        diff_fn = f"{fn}_d{delay}_w{weight}.diff"
    elif output == "one_diff":
        fn_new = "%s_all.new" % fn
        diff_fn = "%s_all.diff" % fn
    else:
        raise NotImplemented

    return diff_fn, fn_new


def mutate_files(files, returns=False, delay=0.5, weight=1.0,
                 output="coin_flip", old_dir=".", big_diff=False):
    old_dir = os.path.abspath(old_dir)
    comby = Comby()
    if returns:
        match = "return :[x];"
        rewrite = "{sleep %f; return :[x];}" % delay
    else:
        raise NotImplemented

    diff_fns = []
    new_fns = []
    #print(files)



    for fn in files:
        diff_fn, new_fn = get_fns(fn, delay, output=output, weight=weight)
        if os.path.isfile(new_fn) and os.path.isfile(diff_fn):
            continue
        with open(fn) as old:

            # source_old = old.read()
            #source_new = comby.rewrite(source_old, match, rewrite,
            #                           language=".cpp")
            #print(type(source_old))
            #print("\n\n\n")
            #print(type(source_new))
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
            #print(source_new)

        diff_fn, new_fn = create_output(fn, source_old, source_new, delay,
                                           new_fn,
                                           diff_fn,
                                           output=output,
                                           weight=weight)
        diff_fns.append(diff_fn)
        new_fns.append(new_fn)
    # Make a temp directory, put all the new_fns in it, then diff the
    # whole thing
    if big_diff:
        bdfn = f'dir_d{delay}_w{weight}.diff'
        big_diff_fn = os.path.join(old_dir, bdfn)
        dir_name = os.path.abspath(os.path.join('.', uuid.uuid4().hex))
        os.makedirs(dir_name)
        # with tempfile.TemporaryDirectory() as dir_name:
        #dir_name = tmpdir.name
        for fn in new_fns:
            shutil.copy(fn, dir_name)
        cmd = f"diff -uN {old_dir} {dir_name}"
        with open(big_diff_fn, 'w') as big_diff:
            print(f"executing cmd: {cmd}")
            subprocess.Popen(shlex.split(cmd), stdout=big_diff)
            print("output to %s" % big_diff_fn)
            print("tempdir: %s" % dir_name)
    #print("about to remove: %s" % dir_name)
    #shutil.rmtree(dir_name)
    #print("removed: %s" % dir_name)

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

    # if output == "one_diff":
    print("calling unified_diff")
    diff = difflib.unified_diff(fromlines, tolines, fn, fn)
    with open(diff_fn, 'w') as diff_file:
        print("writing to %s" % diff_fn)
        diff_file.writelines(diff)


    return diff_fn, fn_new

    #if output == "coin_flip" and weight < 1.0:
    #    diff = difflib.unified_diff(fromlines, tolines, fn, fn)
    #    with open(diff_fn, 'w') as diff_file:
    #        skip_next = False
    #        diff_line_count = 0
    #        print_line_count = 0
    #        to_write = []
    #        for line in diff:
    #            diff_line_count += 1
    #            if not line.startswith("+"):
    #                skip_next = False
    #            if skip_next:
    #                if line.startswith("+"):
    #                    print("not print: %s" % line)
    #                    print_line_count += 1
    #                else:
    #                    print("print    : %s" % line)
    #                    #diff_file.write(line)
    #                    to_write.append(line)
    #                    print_line_count += 1

    #            else:
    #                if coin_flip(weight) or line.startswith("---") or not line.startswith("-"):
    #                    print("print    : %s" % line)
    #                    to_write.append(line)
    #                    #diff_file.write(line)
    #                    print_line_count += 1
    #                else:
    #                    skip_next = True
    #                    print("not print: %s" % line)
    #                    print_line_count += 1
    #        assert(all([x.endswith("\n") for x in to_write]))
    #        diff_file.writelines(to_write)
    #    print("diff_line_count: %s" % diff_line_count)
    #    print("print_line_count: %s" % print_line_count)


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
            delays = [(1/pow(2, x)) for x in range(10)]

            print(weights)
            print(delays)

            #print(files)
            for weight in weights:
                for delay in delays:
                    mutations = mutate_files(files, returns=args.returns,
                                             weight=weight, delay=delay,
                                             output=args.output,
                                             old_dir=args.dir)
        else:
            mutations = mutate_files(files, returns=args.returns,
                                     weight=args.weight, delay=args.delay,
                                     output=args.output, old_dir=args.dir)

if __name__ == '__main__':
    main()
