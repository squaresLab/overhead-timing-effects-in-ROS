#!/usr/bin/env python3
# This is a setup file to set up the overhead timing effects in ROS experiments
# Deby Katz 2019

import shlex
import subprocess
import os.path

def main():

    cmd = "python -m pip install --user pipenv"
    args = shlex.split(cmd)
    subprocess.Popen(args)
    # cmd = "`python -m site --user-base`/bin"
    
    # Build Mavproxy
    DIR_THIS = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(DIR_THIS, "scripts/bin/mavproxy")):
        cmd = "./scripts/build-mavproxy.sh"
        subprocess.Popen(cmd)
    
    # Build the relevant Docker files
    cmd = "make -C ./docker arducopter"
    args = shlex.split(cmd)
    subprocess.Popen(args)
 
    # Create a pipenv
    # Install the requirements
    cmd = "pipenv --python 3.6 shell && pip install -r requirements.txt"
    args = shlex.split(cmd)
    subprocess.Popen(args, shell=True)
    




if __name__ == '__main__':
    main()
