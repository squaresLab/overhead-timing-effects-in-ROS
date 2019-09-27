import argparse
from enum import Enum
from ruamel.yaml import YAML

import roswire


class Mutation:

    def __init__(self, base_version, diff):
        self.base_version = base_version
        self.diff = diff

    def from_dict(d):
        diff = d['diff']
        base_version = BaseVersion(d['base-version'])
        return Mutation(base_version, diff)


class MutationDatabase:
    def __init__(self, contents):
        self.__contents = contents

    def from_file(fn):
        with open(fn, 'r') as f:
            yml = YAML().load(f)
        contents = [Mutation.from_dict(dd) for dd in yml]
        return MutationDatabase(contents)

    def __iter__(self):
        yield from self.__contents


class BaseVersion(Enum):
    Copter_3_6_7 = 'Copter-3.6.7'
    Copter_3_6_6 = 'Copter-3.6.6'
    Copter_3_6_5 = 'Copter-3.6.5'
    Copter_3_6_4 = 'Copter-3.6.4'

    def image(self):
        return 'dskatz/ardu:{}'.format(self.value)

    def snapshot(self):
        name = 'dsk-ardu:{}'.format(self.value)
        coverage_instructions = BugZooCoverageInstructions.from_dict({
            'type': 'gcov',
            'files-to-instrument': [
                'APMrover2/APMrover2.cpp',
                'ArduCopter/ArduCopter.cpp',
                'ArduPlane/ArduPlane.cpp'
            ]
        })
        snapshot = BugZooSnapshot(
            name=name,
            image=self.image,
            dataset='dsk-ardu',
            program='ArduPilot',
            source=None,
            source_dir='/opt/ardupilot',
            languages=[bugzoo.Language.C],
            tests=bugzoo.core.TestSuite([]),
            compiler=bugzoo.compiler.WafCompiler(300),
            instructions_coverage=coverage_instructions)
        return snapshot


def get_diff_fn(args):
    return "../mutations.yml"


def get_docker_image(args):
    return "roswire/example:mavros"


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    rsw = roswire.ROSWire()

    # Read a diff file (generate or existing)
    diff_fn = get_diff_fn(args)

    # Find the docker image
    docker_image = get_docker_image(args)

    # apply the diff file to the docker container
    # try:
    mutation_database = MutationDatabase.from_file(diff_fn)
    # except:
    #    mutation = Mutation(base_version, diff)

    for mutation in mutation_database:
        diff = mutation.diff

        print("\n\nlaunching docker image: %s" % docker_image)

        with rsw.launch(docker_image) as system:
            context = '/ros_ws/src/ArduPilot'

            print("printing diff")
            print(diff)

            print("printing context")
            print(context)

            print("applying patch")
            system.files.patch(context, diff)
            print("patch applied")

            print("rebuilding...")
            dir_workspace = '/ros_ws'

            catkin = system.catkin(dir_workspace)
            catkin.build()
            print("rebuilt")

    # rebuild the docker container
    # run tests via service calls (inputs, with and without instrumentation
    # extract logs via rosbag

    # analyze bag files
    pass


if __name__ == '__main__':
    main()
