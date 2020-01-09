# Create some WPL-style missions

# Deby Katz 2019

import argparse
import attr
import fluffycow
from math import radians, cos, sin, asin, sqrt
import os
import random
import uuid

from ardu import distance_metres, Mission


@attr.s
class Location:
    lat = attr.ib(type=float)
    lon = attr.ib(type=float)
    alt = attr.ib(type=float)


def haversine(loc1, loc2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [loc1.lon, loc1.lat,
                                           loc2.lon, loc2.lat])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    m = 1000 * km
    return m


def generate_waypoint(prev_loc):
    lat = random.gauss(prev_loc.lat, 0.005)
    lon = random.gauss(prev_loc.lon, 0.005)
    alt = random.uniform(0, 10)
    return Location(lat=lat, lon=lon, alt=alt)


def generate_waypoints(home_location, num=8, max_dist=500):
    prev_loc = home_location
    waypoints = []

    gen = fluffycow.factory(Location,
                            lat=fluffycow.gauss(prev_loc.lat, 0.005),
                            lon=fluffycow.gauss(prev_loc.lon, 0.005),
                            alt=fluffycow.uniform(0, 20))
    for i in range(num):
        # Generate a proposed waypoint, based on current waypoint location
        new_waypoint = next(gen)
        # Check if it's too far away. If so, try another one.
        while haversine(new_waypoint, prev_loc) > max_dist:
            print("waypoint out of range")
            new_waypoint = next(gen)
        waypoints.append(new_waypoint)
        prev_loc = new_waypoint
    return waypoints


def make_mission_file(home_location, waypoints, file_location='missions/auto'):
    # make the file location absolute
    if not os.path.isabs(file_location):
        file_location = os.path.abspath(file_location)

    os.makedirs(file_location, exist_ok=True)
    
    # pick a filename
    fn = os.path.join(file_location, "%s.wpl" % uuid.uuid4().hex)    

    mission_file = open(fn, "w")

    # print the header to the file
    mission_file.write("QGC WPL 110\n")

    # print the home location to the file
    index = 0
    current_wp = 0
    frame = 0
    cmd = 16
    hold_time = 0
    accept_radius = 0
    pass_radius = 0
    yaw = 0
    lat = home_location.lat
    lon = home_location.lon
    alt = home_location.alt
    autocontinue = 1
    mission_file.write(f"{index}\t{current_wp}\t{frame}\t{cmd}\t{hold_time}\t" +
                       f"{accept_radius}\t{pass_radius}\t{yaw}\t{lat}\t" +
                       f"{lon}\t{alt}\t{autocontinue}\n")
    cmd = 22
    index = 1
    hold_time = 1
    accept_radius = 5
    alt = random.randrange(2, 20)
    frame = 3
    mission_file.write(f"{index}\t{current_wp}\t{frame}\t{cmd}\t{hold_time}\t" +
                       f"{accept_radius}\t{pass_radius}\t{yaw}\t{lat}\t" +
                       f"{lon}\t{alt}\t{autocontinue}\n")
    accept_radius = 0
    hold_time = 2
    # print the waypoints to the file
    for waypoint in waypoints:
        lat = waypoint.lat
        lon = waypoint.lon
        mission_file.write(f"{index}\t{current_wp}\t{frame}\t{cmd}" +
                           f"\t{hold_time}\t" +
                           f"{accept_radius}\t{pass_radius}\t{yaw}\t{lat}\t" +
                           f"{lon}\t{alt}\t{autocontinue}\n")
        
    index = index + 1
    cmd = 20
    hold_time = 1
    accept_radius = 5
    alt = 0
    mission_file.write(f"{index}\t{current_wp}\t{frame}\t{cmd}\t{hold_time}\t" +
                       f"{accept_radius}\t{pass_radius}\t{yaw}\t{lat}\t" +
                       f"{lon}\t{alt}\t{autocontinue}\n")




def generate_homes(iters=10):
    homes_list = []
    out_of_range_count = 0
    gen = fluffycow.factory(Location,
                            lat=fluffycow.gauss(0.0, 30.0),
                            lon=fluffycow.uniform(-180.0, 180.0),
                            alt=fluffycow.uniform(0.0, 300))
    
    home_lat_gen = fluffycow.gauss(0.0, 30.0)
    home_lon_gen = fluffycow.uniform(-180.0, 180.0)
    for i in range(iters):
        home = next(gen)
        while home.lat > 90 or home.lat < -90:
            print("home_lat %s out of range" % home_lat)
            out_of_range_count = out_of_range_count + 1
            home = next(gen)
        print(home.lat, home.lon)
        homes_list.append(home)
    print("out_of_range_count: %d" % out_of_range_count)
    return homes_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--max_dist', type=int, default=500)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    home_locations = generate_homes(iters=args.num)
    print(home_locations)

    for home_location in home_locations:
        waypoints = generate_waypoints(home_location, max_dist=500)
        print("\n")
        print([(x.lat, x.lon, x.alt) for x in waypoints])
        mission_fn = make_mission_file(home_location, waypoints)
        
if __name__ == "__main__":
    main()
