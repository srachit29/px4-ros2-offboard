#!/usr/bin/env python3
"""
square_mission.py - fly a square with PX4 in offboard mode (ROS 2 Jazzy + px4_msgs).

Sequence:
  1. Stream offboard heartbeats + a hover setpoint above the start point.
  2. Switch PX4 to OFFBOARD mode and arm (retried every second until accepted).
  3. Climb to ALT, then fly the corners of a SIDE x SIDE square,
     holding HOLD_TIME seconds at each corner.
  4. Command LAND and stop once PX4 has disarmed.

Every step is logged to ~/square_log.csv (time, position, setpoint) for plotting.

Frames: PX4 uses NED (x = North, y = East, z = Down), so "5 m up" is z = -5.
"""
import csv
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

# ---------------- mission settings ----------------
ALT = -5.0            # flight altitude in NED [m] (negative = up)
SIDE = 5.0            # square side length [m]
ACCEPT_RADIUS = 0.3   # how close counts as "reached" [m]
HOLD_TIME = 2.0       # hover time at each corner [s]
RATE_HZ = 10.0        # setpoint rate (PX4 needs > 2 Hz)

# Square corners relative to the start point (North, East)
CORNERS = [(0.0, 0.0), (SIDE, 0.0), (SIDE, SIDE), (0.0, SIDE), (0.0, 0.0)]

# ---------------- topic names (match your PX4 version) ----------------
TOPIC_OFFBOARD = '/fmu/in/offboard_control_mode'
TOPIC_SETPOINT = '/fmu/in/trajectory_setpoint'
TOPIC_COMMAND = '/fmu/in/vehicle_command_v1'
TOPIC_LOCAL_POS = '/fmu/out/vehicle_local_position_v1'
TOPIC_STATUS = '/fmu/out/vehicle_status_v5'


class SquareMission(Node):
    def __init__(self):
        super().__init__('square_mission')

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(OffboardControlMode, TOPIC_OFFBOARD, pub_qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, TOPIC_SETPOINT, pub_qos)
        self.command_pub = self.create_publisher(VehicleCommand, TOPIC_COMMAND, pub_qos)

        self.create_subscription(VehicleLocalPosition, TOPIC_LOCAL_POS, self.on_position, sub_qos)
        self.create_subscription(VehicleStatus, TOPIC_STATUS, self.on_status, sub_qos)

        self.pos = None          # latest VehicleLocalPosition
        self.status = None       # latest VehicleStatus
        self.home = None         # (x, y) at start
        self.state = 'INIT'      # INIT -> CLIMB -> SQUARE -> LAND -> DONE
        self.corner = 0
        self.reached_since = None
        self.tick = 0
        self.done = False
        self.t0 = self.now()

        log_path = os.path.expanduser('~/square_log.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.log = csv.writer(self.log_file)
        self.log.writerow(['t', 'state', 'x', 'y', 'z', 'sp_x', 'sp_y', 'sp_z'])

        self.create_timer(1.0 / RATE_HZ, self.step)
        self.get_logger().info(f'Square mission ready. Logging to {log_path}')

    # ---------------- helpers ----------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def stamp_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def on_position(self, msg):
        self.pos = msg

    def on_status(self, msg):
        self.status = msg

    def is_offboard(self):
        return self.status is not None and \
            self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def is_armed(self):
        return self.status is not None and \
            self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def send_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(p1)
        msg.param2 = float(p2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.stamp_us()
        self.command_pub.publish(msg)

    def publish_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.stamp_us()
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = 0.0  # face North
        msg.timestamp = self.stamp_us()
        self.setpoint_pub.publish(msg)

    def corner_target(self, i):
        dx, dy = CORNERS[i]
        return self.home[0] + dx, self.home[1] + dy, ALT

    def distance_to(self, target):
        return math.sqrt((self.pos.x - target[0]) ** 2 +
                         (self.pos.y - target[1]) ** 2 +
                         (self.pos.z - target[2]) ** 2)

    # ---------------- main loop (10 Hz) ----------------
    def step(self):
        self.tick += 1
        if self.state not in ('LAND', 'DONE'):
            self.publish_heartbeat()

        if self.pos is None:
            if self.tick % 20 == 0:
                self.get_logger().info('Waiting for position data from PX4...')
            return

        if self.home is None:
            self.home = (self.pos.x, self.pos.y)
            self.get_logger().info(f'Home set at x={self.home[0]:.2f}, y={self.home[1]:.2f}')

        target = None
        t = self.now()

        if self.state == 'INIT':
            target = self.corner_target(0)
            self.publish_setpoint(*target)
            # after 1 s of setpoints, request OFFBOARD + ARM once per second until accepted
            if self.tick > RATE_HZ and self.tick % int(RATE_HZ) == 0:
                if not self.is_offboard():
                    self.get_logger().info('Requesting OFFBOARD mode...')
                    self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                if not self.is_armed():
                    self.get_logger().info('Requesting ARM...')
                    self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.is_offboard() and self.is_armed():
                    self.get_logger().info(f'Armed in OFFBOARD. Climbing to {-ALT:.1f} m')
                    self.state = 'CLIMB'

        elif self.state in ('CLIMB', 'SQUARE'):
            if not self.is_offboard():
                self.get_logger().warn('Left OFFBOARD mode (pilot/failsafe). Stopping mission.')
                self.state = 'DONE'
                self.done = True
                return

            target = self.corner_target(self.corner)
            self.publish_setpoint(*target)

            if self.distance_to(target) < ACCEPT_RADIUS:
                if self.reached_since is None:
                    self.reached_since = t
                    self.get_logger().info(
                        f'Reached corner {self.corner} '
                        f'({CORNERS[self.corner][0]:.0f}, {CORNERS[self.corner][1]:.0f})')
                elif t - self.reached_since > HOLD_TIME:
                    self.reached_since = None
                    self.corner += 1
                    self.state = 'SQUARE'
                    if self.corner >= len(CORNERS):
                        self.get_logger().info('Square complete. Landing.')
                        self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                        self.state = 'LAND'
            else:
                self.reached_since = None

        elif self.state == 'LAND':
            if self.status is not None and not self.is_armed():
                self.get_logger().info('Landed and disarmed. Mission finished.')
                self.state = 'DONE'
                self.done = True

        # log
        sp = target if target is not None else (float('nan'),) * 3
        self.log.writerow([f'{t - self.t0:.2f}', self.state,
                           f'{self.pos.x:.3f}', f'{self.pos.y:.3f}', f'{self.pos.z:.3f}',
                           f'{sp[0]:.3f}', f'{sp[1]:.3f}', f'{sp[2]:.3f}'])

    def close(self):
        self.log_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = SquareMission()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
