"""Find the local interface used to reach an explicitly configured robot IP."""
import os
import socket


def get_local_ip(ip):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.connect((ip, 80))
        return connection.getsockname()[0]


def get_ip(robot_ip=None):
    robot_ip = robot_ip or os.environ.get("FLEXIV_ROBOT_IP")
    if not robot_ip:
        raise ValueError("Pass robot_ip or set FLEXIV_ROBOT_IP")
    return robot_ip, get_local_ip(robot_ip)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_ip", nargs="?", help="Robot IP (defaults to FLEXIV_ROBOT_IP)")
    args = parser.parse_args()
    print(*get_ip(args.robot_ip))
