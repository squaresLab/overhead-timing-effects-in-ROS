import argparse
import roswire


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', action='append', type=str,
                        help='Specify the files to instrument')
    parser.add_argument('--patches', action='append', type=str,
                        help='Specify the patches to apply')
    parser.add_argument('--image', type=str,
                        default='/roswire/example:mavros')
    parser.add_argument('--context', type=str,
                        default='/ros_ws/src/ArduPilot/')
    args = parser.parse_args()
    return args


def apply_one_patch(patch_fn, rsw):
    with open(patch_fn) as f:
       diff = f.read()

    # we use 'launch' to create a temporary container for the application
    # when the context is closed, either by reaching the end of the with
    # block or by abruptly encountering an exception, the container will be
    # automatically destroyed.
    with rsw.launch('roswire/example:mavros') as system:
       print("applying patch...")
       context = '/ros_ws/src/ArduPilot/'
       system.files.patch(context, diff)
       print("patch applied")

       # rebuild via catkin tools
       print("rebuilding...")
       dir_workspace = '/ros_ws'
       catkin = system.catkin(dir_workspace)
       catkin.build()
       print("rebuilt")


def main():
    rsw = roswire.ROSWire()

    args = parse_args()

    for patch_fn in args.patches:
        apply_one_patch(patch_fn, rsw)


if __name__ == '__main__':
    main()
