#!/usr/bin/env python3
"""
pd_square.py - fly the 5 x 5 m square with YOUR OWN position controller (PD).

Difference to square_mission.py:
  square_mission.py sends POSITION setpoints -> PX4's position controller flies.
  pd_square.py computes an ACCELERATION command itself -> PX4 only handles
  tilting and motors. The position control loop is now yours.

Control law (per axis, NED frame):
    a_cmd = a_ref + Kp * (p_ref - p) + Kd * (v_ref - v)

    p, v          measured position / velocity  (from VehicleLocalPosition)
    p_ref, v_ref  where the drone should be / how fast it should move now
    a_ref         feed-forward acceleration of the reference (0 on straight lines)
    Kp            "spring": pulls the drone towards the reference
    Kd            "damper": brakes, prevents overshoot

Gains come from a 2nd-order system:  Kp = wn^2,  Kd = 2 * zeta * wn
    wn   = natural frequency (higher = faster, more aggressive)
    zeta = damping ratio (1.0 = no overshoot, < 1 = some overshoot)

Phases:
  INIT/CLIMB  PX4 position control takes off and climbs to 5 m (same as before)
  SQUARE      your PD controller tracks a reference moving along the square
  LAND        PX4 lands

Log: ~/pd_square_log.csv
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
ALT = -5.0            # altitude in NED [m] (negative = up)
SIDE = 5.0            # square side [m]
SPEED = 1.25          # reference speed along each side [m/s]
HOLD_TIME = 2.0       # pause at each corner [s]
RATE_HZ = 50.0        # control rate [Hz] (faster than before: we close the loop now)
CORNERS = [(0.0, 0.0), (SIDE, 0.0), (SIDE, SIDE), (0.0, SIDE), (0.0, 0.0)]

# ---------------- PD gains ----------------
WN_XY, ZETA_XY = 1.5, 0.9     # horizontal
WN_Z, ZETA_Z = 2.0, 0.9       # vertical
KP_XY, KD_XY = WN_XY ** 2, 2 * ZETA_XY * WN_XY
KP_Z, KD_Z = WN_Z ** 2, 2 * ZETA_Z * WN_Z

A_MAX_XY = 3.0        # acceleration limits [m/s^2]
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
    """Reference that moves along the square at constant speed, pausing at corners."""

    def __init__(self, home):
        self.segments = []   # (start_time, end_time, start_point, end_point)
        t = HOLD_TIME
        for i in range(len(CORNERS) - 1):
            a = (home[0] + CORNERS[i][0], home[1] + CORNERS[i][1])
            b = (home[0] + CORNERS[i + 1][0], home[1] + CORNERS[i + 1][1])
            duration = math.hypot(b[0] - a[0], b[1] - a[1]) / SPEED
            self.segments.append((t, t + duration, a, b))
            t += duration + HOLD_TIME
        self.end_time = t

    def at(self, t):
        """Return (p_ref, v_ref) at time t since the square started."""
        first = self.segments[0]
        if t < first[0]:
            return (first[2][0], first[2][1], ALT), (0.0, 0.0, 0.0)
        for i, (t0, t1, a, b) in enumerate(self.segments):
            if t0 <= t <= t1:
                s = (t - t0) / (t1 - t0)
                p = (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), ALT)
                v = ((b[0] - a[0]) / (t1 - t0), (b[1] - a[1]) / (t1 - t0), 0.0)
                return p, v
            next_start = self.segments[i + 1][0] if i + 1 < len(self.segments) else self.end_time
            if t1 < t < next_start:   # holding at corner b
                return (b[0], b[1], ALT), (0.0, 0.0, 0.0)
        last = self.segments[-1][3]
        return (last[0], last[1], ALT), (0.0, 0.0, 0.0)


class PDSquare(Node):
    def __init__(self):
        super().__init__('pd_square')

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

        log_path = os.path.expanduser('~/pd_square_log.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.log = csv.writer(self.log_file)
        self.log.writerow(['t', 'state', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                           'ref_x', 'ref_y', 'ref_z', 'ax', 'ay', 'az'])

        self.create_timer(1.0 / RATE_HZ, self.step)
        self.get_logger().info(
            f'PD square ready. Kp_xy={KP_XY:.2f} Kd_xy={KD_XY:.2f} '
            f'Kp_z={KP_Z:.2f} Kd_z={KD_Z:.2f}. Logging to {log_path}')

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
        """Tell PX4 which setpoint type we send: position (PX4 controls) or acceleration (we control)."""
        msg = OffboardControlMode()
        msg.position = not acceleration
        msg.velocity = False
        msg.acceleration = acceleration
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.stamp_us()
        self.offboard_pub.publish(msg)

    def publish_position(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = 0.0
        msg.timestamp = self.stamp_us()
        self.setpoint_pub.publish(msg)

    def publish_acceleration(self, ax, ay, az):
        msg = TrajectorySetpoint()
        msg.position = [NAN, NAN, NAN]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [float(ax), float(ay), float(az)]
        msg.yaw = 0.0
        msg.timestamp = self.stamp_us()
        self.setpoint_pub.publish(msg)

    # ---------------- the controller ----------------
    def pd_control(self, p_ref, v_ref):
        p = (self.pos.x, self.pos.y, self.pos.z)
        v = (self.pos.vx, self.pos.vy, self.pos.vz)
        ax = KP_XY * (p_ref[0] - p[0]) + KD_XY * (v_ref[0] - v[0])
        ay = KP_XY * (p_ref[1] - p[1]) + KD_XY * (v_ref[1] - v[1])
        az = KP_Z * (p_ref[2] - p[2]) + KD_Z * (v_ref[2] - v[2])
        # limit horizontal acceleration as a vector (keeps the direction)
        a_xy = math.hypot(ax, ay)
        if a_xy > A_MAX_XY:
            ax, ay = ax * A_MAX_XY / a_xy, ay * A_MAX_XY / a_xy
        return ax, ay, clamp(az, A_MAX_Z)

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
            self.publish_position(*ref_p)

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
                    self.ref = SquareReference(self.home)
                    self.square_t0 = t
                    self.state = 'SQUARE'
                    self.get_logger().info('At start. Switching to YOUR PD controller.')

        elif self.state == 'SQUARE':
            if not self.is_offboard():
                self.get_logger().warn('Left OFFBOARD mode (pilot/failsafe). Stopping.')
                self.done = True
                return
            ts = t - self.square_t0
            ref_p, ref_v = self.ref.at(ts)
            acc = self.pd_control(ref_p, ref_v)
            self.publish_mode(acceleration=True)
            self.publish_acceleration(*acc)
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
    node = PDSquare()
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
