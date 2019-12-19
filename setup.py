# This is a setup file to set up the overhead timing effects in ROS experiments
# Deby Katz 2019

import shlex
import subprocess

def main():

    cmd = "python -m pip install --user pipenv"
    args = shlex.split(cmd)
    subprocess.Popen(args)
    # cmd = "`python -m site --user-base`/bin"
    
    # Build Mavproxy
    cmd = "./scripts/build-mavproxy.sh"
    subprocess.Popen(cmd)
    
    # Build the relevant Docker files
    cmd = "make -C ./docker arducopter"
    args = shlex.split(cmd)
    subprocess.Popen(args)
 
    # Create a pipenv
    # Install the requirements
    cmd = "pipenv --python 3.6 && pip install -r requirements.txt"
    args = shlex.split(cmd)
    subprocess.Popen(args, shell=True)
    




if __name__ == '__main__':
    main()
