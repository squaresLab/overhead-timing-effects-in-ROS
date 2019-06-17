# overhead-timing-effects-in-ROS


## Setup

To setup the environment in which to run the scripts, first create a virtual
environment (ideally via `pipenv`), and from within that environment, execute
the following:

```
(timing-env) $ pip install -r requirements.txt
```

To build the example Docker image for MAVROS:

```
$ cd docker
$ make mavros
```
