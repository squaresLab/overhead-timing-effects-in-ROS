# -*- coding: utf-8 -*-
"""
This module provides interfaces and data structures for interacting with
ArduPilot via Dronekit and MAVLink.
"""
__all__ = ('Mission',)

from typing import Sequence, Tuple, Optional, Iterator, FrozenSet
import os
import time
import math
import contextlib
import signal
import subprocess
import pkg_resources
import logging

import attr
import dronekit
import roswire
from bugzoo import Container as BugZooContainer
from roswire.util import Stopwatch
from roswire.proxy.container import ShellProxy as ROSWireShell

logger = logging.getLogger(__name__)  # type: logging.Logger
logger.setLevel(logging.DEBUG)


def distance_metres(x: dronekit.LocationGlobal, y: dronekit.LocationGlobal) -> float:
    """Computes the ground distance in metres between two locations.

    This method is an approximation, and will not be accurate over large
    distances and close to the earth's poles.

    Source
    ------
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    d_lat = y.lat - x.lat
    d_lon = y.lon - x.lon
    return math.sqrt((d_lat*d_lat) + (d_lon*d_lon)) * 1.113195e5


@attr.s(frozen=True, slots=True)
class Mission(Sequence[dronekit.Command]):
    """Represents a WPL mission."""
    filename: str = attr.ib(converter=os.path.abspath)
    commands: Sequence[dronekit.Command] = \
        attr.ib(repr=False, converter=tuple, hash=False, eq=False)
    home_location: Tuple[float, float, float, float] = \
        attr.ib(init=False, repr=False, eq=False, hash=False)
    waypoints: FrozenSet[int] = \
        attr.ib(init=False, repr=False, eq=False, hash=False)

    def __attrs_post_init__(self) -> None:
        cmd_home = self.commands[0]
        home_lat = float(cmd_home.x)
        home_lon = float(cmd_home.y)
        home_alt = float(cmd_home.z)
        home_heading = 0.0
        object.__setattr__(self, 'home_location',
                           (home_lat, home_lon, home_alt, home_heading))
        object.__setattr__(self, 'waypoints',
                           frozenset(i for i in range(len(self.commands))))

    @commands.validator
    def validate_commands(self, attr, value):
        if not self.commands:
            raise ValueError("mission must have at least one command")

    @classmethod
    def from_file(cls, fn: str) -> 'Mission':
        logger.debug("loading mission file: %s", fn)
        with open(fn, 'r') as f:
            lines = [l.strip() for l in f]
        return Mission(filename=fn,
                       commands=[cls.line_to_command(l) for l in lines[1:]])

    @staticmethod
    def line_to_command(s: str) -> dronekit.Command:
        """Parses a WPL line to a Dronekit command."""
        logger.debug("parsing command string: %s", s)
        args = s.split()
        arg_index = int(args[0])
        arg_currentwp = 0 # int(args[1])
        arg_frame = int(args[2])
        arg_cmd = int(args[3])
        arg_autocontinue = 0 # not supported by dronekit
        (p1, p2, p3, p4, x, y, z) = [float(x) for x in args[4:11]]
        cmd = dronekit.Command(
            0, 0, 0, arg_frame, arg_cmd, arg_currentwp, arg_autocontinue,
            p1, p2, p3, p4, x, y, z)
        return cmd

    def __getitem__(self, index: int) -> dronekit.Command:
        """Retrieves the i-th command from this mission."""
        return self.commands[index]

    def __len__(self) -> int:
        """Returns the number of commands in this mission."""
        return len(self.commands)

    def issue(self,
              connection: dronekit.Vehicle,
              *,
              timeout: float = 30.0
              ) -> None:
        """Issues this mission to a given vehicle.

        Parameters
        ----------
        connection: dronekit.Vehicle
            A connection to the vehicle.
        timeout: float
            A timeout that is enforced on the process of preparing the vehicle
            and issuing this mission.

        Raises
        ------
        TimeoutError
            If the timeout occurs before the mission has been issued to the
            vehicle.
        """
        timer = Stopwatch()
        timer.start()

        def time_left() -> float:
            return max(0.0, timeout - timer.duration)

        logger.debug("waiting for vehicle to be armable")
        while not connection.is_armable:
            if timer.duration > timeout:
                raise TimeoutError
            time.sleep(0.05)

        # set home location
        logger.debug("waiting for home location")
        while not connection.home_location:
            if timer.duration > timeout:
                raise TimeoutError
            vcmds = connection.commands
            vcmds.download()
            try:
                vcmds.wait_ready(timeout=time_left())
            except dronekit.TimeoutError:
                raise TimeoutError
            time.sleep(0.1)
        logger.debug("determined home location: %s", connection.home_location)

        logger.debug("attempting to arm vehicle")
        connection.armed = True
        while not connection.armed:
            if timer.duration > timeout:
                raise TimeoutError
            time.sleep(0.05)
        logger.debug("armed vehicle")

        # upload mission
        vcmds = connection.commands
        vcmds.clear()
        for command in self:
            vcmds.add(command)
        vcmds.upload()
        try:
            vcmds.wait_ready(timeout=time_left())
        except dronekit.TimeoutError:
            raise TimeoutError

        # start mission
        # FIXME check that mode was changed!
        logger.debug("switching to AUTO mode")
        connection.mode = dronekit.VehicleMode('AUTO')
        logger.debug("switched to AUTO mode")
        message = connection.message_factory.command_long_encode(
            0, 0, 300, 0, 1, len(self) + 1, 0, 0, 0, 0, 4)
        connection.send_mavlink(message)

    def execute(self,
                connection: dronekit.Vehicle,
                *,
                timeout_setup: float = 90.0,
                timeout_mission: float = 120.0
                ) -> None:
        """Executes this mission on a given vehicle.

        Parameters
        ----------
        connection: dronekit.Vehicle
            A connection to the vehicle.
        timeout_setup: float
            The timeout to enforce during mission setup.
        timeout_mission: float
            The timeout to enforce during the execution of the mission itself
            (i.e., after setup has been performed).
        """
        self.issue(connection, timeout=timeout_setup)

        timer = Stopwatch()
        timer.start()
        has_started = False
        command_num = 0

        def listener_statustext(other, name, message):
            logger.debug('STATUSTEXT: %s', message.text)

        logger.debug('attaching STATUSTEXT listener')
        connection.add_message_listener('STATUSTEXT', listener_statustext)
        logger.debug('attached STATUSTEXT listener')

        def distance_to_home():
            return distance_metres(connection.home_location, connection.location.global_frame)

        while True:
            command_last = command_num
            command_num = connection.commands.next
            has_started |= command_num != 0

            # if we've reached the next WP, print a summary of the copter state
            if command_num != command_last:
                logger.debug("NEXT WP: %d", command_num)
                logger.debug("HOME: %s", connection.home_location)
                logger.debug("MODE: %s", connection.mode.name)
                logger.debug("LOCATION: %s", connection.location.global_frame)
                logger.debug("DISTANCE TO HOME: %.2f metres", distance_to_home())

            # the command pointer rolls back to zero upon mission completion
            if has_started and command_num == 0:
                break

            if timer.duration > timeout_mission:
                logger.debug("timeout occurred during mission execution.")
                raise TimeoutError("mission did not complete before timeout")

            time.sleep(0.05)

        logger.debug('removing STATUSTEXT listener')
        connection.add_message_listener('STATUSTEXT', listener_statustext)
        logger.debug('removed STATUSTEXT listener')
        logger.debug("mission terminated")
