#!/usr/bin/env python3
"""
tracking_square.py - fly the 5 x 5 m square with your own controller.

Two choices, set as ROS parameters:

  controller:=pd   hand-tuned PD gains          (same as pd_square.py)
  controller:=lqr  gains from LQR (optimal for the chosen weights Q, R)

  reference:=linear  target moves at constant speed and stops instantly at corners
  reference:=smooth  minimum-jerk target: speeds up and slows down smoothly,
                     and also gives the controller the planned acceleration
                     (feed-forward), so it does not have to wait for an error

Example:
  ros2 run px4_offboard tracking_square --ros-args -p controller:=lqr -p reference:=smooth

Control law (per axis, NED):
    a_cmd = a_ref + K_p * (p_ref - p) + K_d * (v_ref - v)

LQR for one axis of a point mass (double integrator):
    state x = [position error, velocity error],  x_dot = A x + B u
    A = [[0, 1], [0, 0]],  B = [[0], [1]],  u = acceleration
    minimise  J = integral( q_p * e_p^2 + q_v * e_v^2 + r * u^2 ) dt
    The Riccati equation has a closed-form solution here:
        K_p = sqrt(q_p / r)
        K_d = sqrt(q_v / r + 2 * K_p)

Log: ~/square_<controller>_<reference>.csv
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

# ---------------- mission ----------------
ALT = -5.0             # altitude in NED [m] (negative = up)
SIDE = 5.0             # square side [m]
SEG_TIME = 4.0         # time per side [s] (same for both references -> fair comparison)
HOLD_TIME = 2.0        # pause at each corner [s]
RATE_HZ = 50.0
CORNERS = [(0.0, 0.0), (SIDE, 0.0), (SIDE, SIDE), (0.0, SIDE), (0.0, 0.0)]

# ---------------- controllers ----------------
# PD: hand-picked from natural frequency and damping
PD_GAINS = {
    'xy': (1.5 ** 2, 2 * 0.9 * 1.5),     # Kp = wn^2, Kd = 2*zeta*wn
    'z': (2.0 ** 2, 2 * 0.9 * 2.0),
}

# LQR: weights -> gains (closed-form for a double integrator)
LQR_WEIGHTS = {
    #       q_p   q_v   r
    'xy': (9.0, 1.0, 1.0),
    'z': (16.0, 1.0, 1.0),
}


def lqr_gains(q_p, q_v, r):
    k_p = math.sqrt(q_p / r)
    k_d = math.sqrt(q_v / r + 2.0 * k_p)
    return k_p, k_d


A_MAX_XY = 3.0
A_MAX_Z = 2.0

# ---------------- topics (match your PX4 version) ----------------
TOPIC_OFFBOARD = '/fmu/in/offboard_control_mode'
TOPIC_SETPOINT = '/fmu/in/trajectory_setpoint'
TOPIC_COMMAND = '/fmu/in/vehicle_command_v1'
TOPIC_LOCAL_POS = '/fmu/out/vehicle_local_position_v1'
TOPIC_STATUS = '/fmu/out/vehicle_status_v5'

NAN = float('nan')


def clamp(value, limit):
    return max(-limit, min(limit, value))


class SquareReference:
    """Moving target along the square.

    linear: constant speed, instant stop at corners (a_ref = 0)
    smooth: minimum-jerk profile s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,
            zero speed and zero acceleration at both ends of every side.
    """

    def __init__(self, home, smooth):
        self.smooth = smooth
        self.segments = []
        t = HOLD_TIME
        for i in range(len(CORNERS) - 1):
            a = (home[0] + CORNERS[i][0], home[1] + CORNERS[i][1])
            b = (home[0] + CORNERS[i + 1][0], home[1] + CORNERS[i + 1][1])
            self.segments.append((t, t + SEG_TIME, a, b))
            t += SEG_TIME + HOLD_TIME
        self.end_time = t

    def profile(self, tau, T):
        """Return progress s, ds/dt, d2s/dt2 for normalised time tau in [0, 1]."""
        if not self.smooth:
            return tau, 1.0 / T, 0.0
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
        dds = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / T ** 2
        return s, ds, dds

    def at(self, t):
        """Return p_ref, v_ref, a_ref at time t since the square started."""
        zero = (0.0, 0.0, 0.0)
        point = self.segments[0][2]
        for t0, t1, a, b in self.segments:
            if t < t0:
                break
            point = b
            if t <= t1:
                s, ds, dds = self.profile((t - t0) / (t1 - t0), t1 - t0)
                d = (b[0] - a[0], b[1] - a[1])
                p = (a[0] + s * d[0], a[1] + s * d[1], ALT)
                v = (ds * d[0], ds * d[1], 0.0)
                acc = (dds * d[0], dds * d[1], 0.0)
                return p, v, acc
        return (point[0], point[1], ALT), zero, zero


class TrackingSquare(Node):
    def __init__(self):
        super().__init__('tracking_square')
        self.controller = self.declare_parameter('controller', 'lqr').value
        self.reference = self.declare_parameter('reference', 'smooth').value
        if self.controller not in ('pd', 'lqr') or self.reference not in ('linear', 'smooth'):
            raise ValueError('controller must be pd|lqr, reference must be linear|smooth')

        if self.controller == 'pd':
            self.gains = PD_GAINS
        else:
            self.gains = {axis: lqr_gains(*w) for axis, w in LQR_WEIGHTS.items()}

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.offboard_pub = self.create_publisher(OffboardControlMode, TOPIC_OFFBOARD, pub_qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, TOPIC_SETPOINT, pub_qos)
        self.command_pub = self.create_publisher(VehicleCommand, TOPIC_COMMAND, pub_qos)
        self.create_subscription(VehicleLocalPosition, TOPIC_LOCAL_POS, self.on_position, sub_qos)
        self.create_subscription(VehicleStatus, TOPIC_STATUS, self.on_status, sub_qos)

        self.pos = None
        self.status = None
        self.home = None
        self.ref = None
        self.square_t0 = None
        self.state = 'INIT'
        self.tick = 0
        self.done = False
        self.t0 = self.now()

        log_path = os.path.expanduser(f'~/square_{self.controller}_{self.reference}.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.log = csv.writer(self.log_file)
        self.log.writerow(['t', 'state', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                           'ref_x', 'ref_y', 'ref_z', 'ax', 'ay', 'az'])

        self.create_timer(1.0 / RATE_HZ, self.step)
        (kp_xy, kd_xy), (kp_z, kd_z) = self.gains['xy'], self.gains['z']
        self.get_logger().info(
            f'controller={self.controller} reference={self.reference} | '
            f'Kp_xy={kp_xy:.2f} Kd_xy={kd_xy:.2f} Kp_z={kp_z:.2f} Kd_z={kd_z:.2f} | '
            f'log: {log_path}')

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

    def publish_mode(self, acceleration):
        msg = OffboardControlMode()
        msg.position = not acceleration
        msg.velocity = False
        msg.acceleration = acceleration
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.stamp_us()
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, position=None, acceleration=None):
        msg = TrajectorySetpoint()
        msg.position = [float(v) for v in position] if position else [NAN] * 3
        msg.velocity = [NAN] * 3
        msg.acceleration = [float(v) for v in acceleration] if acceleration else [NAN] * 3
        msg.yaw = 0.0
        msg.timestamp = self.stamp_us()
        self.setpoint_pub.publish(msg)

    # ---------------- the controller ----------------
    def control(self, p_ref, v_ref, a_ref):
        p = (self.pos.x, self.pos.y, self.pos.z)
        v = (self.pos.vx, self.pos.vy, self.pos.vz)
        a = []
        for i in range(3):
            kp, kd = self.gains['z' if i == 2 else 'xy']
            a.append(a_ref[i] + kp * (p_ref[i] - p[i]) + kd * (v_ref[i] - v[i]))
        a_xy = math.hypot(a[0], a[1])
        if a_xy > A_MAX_XY:
            a[0], a[1] = a[0] * A_MAX_XY / a_xy, a[1] * A_MAX_XY / a_xy
        a[2] = clamp(a[2], A_MAX_Z)
        return tuple(a)

    # ---------------- main loop ----------------
    def step(self):
        self.tick += 1
        if self.pos is None:
            if self.tick % int(RATE_HZ * 2) == 0:
                self.get_logger().info('Waiting for position data from PX4...')
            return
        if self.home is None:
            self.home = (self.pos.x, self.pos.y)
            self.get_logger().info(f'Home set at x={self.home[0]:.2f}, y={self.home[1]:.2f}')

        t = self.now()
        ref_p, acc = (NAN, NAN, NAN), (NAN, NAN, NAN)

        if self.state in ('INIT', 'CLIMB'):
            self.publish_mode(acceleration=False)
            ref_p = (self.home[0], self.home[1], ALT)
            self.publish_setpoint(position=ref_p)

            if self.state == 'INIT' and self.tick > RATE_HZ and self.tick % int(RATE_HZ) == 0:
                if not self.is_offboard():
                    self.get_logger().info('Requesting OFFBOARD mode...')
                    self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                if not self.is_armed():
                    self.get_logger().info('Requesting ARM...')
                    self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.is_offboard() and self.is_armed():
                    self.get_logger().info('Armed. PX4 climbs to start point...')
                    self.state = 'CLIMB'

            if self.state == 'CLIMB':
                dist = math.dist((self.pos.x, self.pos.y, self.pos.z), ref_p)
                if dist < 0.2 and math.hypot(self.pos.vx, self.pos.vy) < 0.2:
                    self.ref = SquareReference(self.home, self.reference == 'smooth')
                    self.square_t0 = t
                    self.state = 'SQUARE'
                    self.get_logger().info(
                        f'At start. Switching to YOUR {self.controller.upper()} controller.')

        elif self.state == 'SQUARE':
            if not self.is_offboard():
                self.get_logger().warn('Left OFFBOARD mode (pilot/failsafe). Stopping.')
                self.done = True
                return
            ts = t - self.square_t0
            ref_p, ref_v, ref_a = self.ref.at(ts)
            acc = self.control(ref_p, ref_v, ref_a)
            self.publish_mode(acceleration=True)
            self.publish_setpoint(acceleration=acc)
            if ts > self.ref.end_time:
                self.get_logger().info('Square complete. Landing.')
                self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.state = 'LAND'

        elif self.state == 'LAND':
            if self.status is not None and not self.is_armed():
                self.get_logger().info('Landed and disarmed. Mission finished.')
                self.done = True

        self.log.writerow([f'{t - self.t0:.3f}', self.state,
                           f'{self.pos.x:.3f}', f'{self.pos.y:.3f}', f'{self.pos.z:.3f}',
                           f'{self.pos.vx:.3f}', f'{self.pos.vy:.3f}', f'{self.pos.vz:.3f}',
                           f'{ref_p[0]:.3f}', f'{ref_p[1]:.3f}', f'{ref_p[2]:.3f}',
                           f'{acc[0]:.3f}', f'{acc[1]:.3f}', f'{acc[2]:.3f}'])

    def close(self):
        self.log_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingSquare()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
