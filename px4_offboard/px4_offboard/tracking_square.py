#!/usr/bin/env python3
"""
tracking_square.py - fly the 5 x 5 m square with your own controller.

Two choices, set as ROS parameters:

  controller:=pd       hand-tuned PD gains          (same as pd_square.py)
  controller:=lqr      LQR on a point-mass model (position, velocity)
  controller:=mpc      Model Predictive Control (mpc.py): plans 1.5 s ahead with a
                       model that includes the drone's lag and delay, and uses the
                       FUTURE reference (it sees corners coming)
  controller:=lqr_lag  LQR on a model that also includes how slowly the drone
                       follows an acceleration command (time constant tau),
                       and feeds back the measured acceleration as well

LQR weights can be changed without editing code:
  -p q_p:=9.0 -p q_v:=4.0 -p q_a:=0.1 -p r:=1.0 -p tau:=0.2

  reference:=linear  target moves at constant speed and stops instantly at corners
  reference:=smooth  minimum-jerk target: speeds up and slows down smoothly,
                     and also gives the controller the planned acceleration
                     (feed-forward), so it does not have to wait for an error

  trajectory:=square   5 x 5 m square (default)
  trajectory:=figure8  figure-eight, 6 m long, 3 m wide, 12 s per lap, flown twice
  controller:=px4      baseline: PX4's own position controller follows the same reference
                       (position + velocity + acceleration feed-forward)

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

Lag model used by lqr_lag (per horizontal axis), measured with tools/fit_lag.py:
    a_actual' = (a_cmd - a_actual) / tau,   tau ~ 0.2 s in PX4 SITL
    state x = [e_p, e_v, e_a],  gains from the discrete Riccati equation (numpy)

Log: ~/square_<controller>_<reference><tag>.csv
"""
import csv
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from collections import deque

from px4_offboard.mpc import AxisMPC, predict_through_delay

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


def dlqr(A, B, Q, R, iters=10000):
    """Discrete-time LQR by iterating the Riccati equation (numpy only)."""
    import numpy as np
    P = Q.copy()
    for _ in range(iters):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(P_next - P)) < 1e-10:
            break
        P = P_next
    return K


def lqr_lag_gains(q_p, q_v, q_a, r, tau, dt):
    """Gains [K_p, K_v, K_a] for the model p' = v, v' = a, a' = (u - a) / tau."""
    import numpy as np
    A_c = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / tau]])
    B_c = np.array([[0.0], [0.0], [1.0 / tau]])
    A = np.eye(3) + A_c * dt          # simple Euler discretisation, dt = control period
    B = B_c * dt
    K = dlqr(A, B, np.diag([q_p, q_v, q_a]), np.array([[r]]))
    return tuple(float(k) for k in K[0])


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


class FigureEightReference:
    """Figure-eight (lemniscate of Gerono) at constant height, flown twice.

    shape:  north = A sin(phi),  east = (A / 2) sin(2 phi)     (passes through home at phi = 0)
    phase:  phi_dot = w * sigma(t), sigma ramps 0 -> 1 -> 0 with minimum-jerk profiles,
            so the drone starts and stops smoothly at home.
    Returns position, velocity AND acceleration (feed-forward), like the smooth square.
    """

    def __init__(self, home, size=3.0, period=12.0, laps=2, ramp=3.0):
        self.home, self.A, self.w, self.Tr = home, size, 2 * math.pi / period, ramp
        self.t_start = HOLD_TIME
        self.T_cruise = laps * period - ramp          # each ramp adds ramp/2 of full-speed time
        self.t_ramp_down = self.t_start + ramp + self.T_cruise
        self.t_stop = self.t_ramp_down + ramp
        self.end_time = self.t_stop + HOLD_TIME

    def _phase(self, t):
        """Return phi, phi_dot, phi_ddot at time t."""
        w, Tr = self.w, self.Tr
        if t <= self.t_start:
            return 0.0, 0.0, 0.0
        if t < self.t_start + Tr:                       # ramp up
            tau = (t - self.t_start) / Tr
            sig = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
            dsig = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / Tr
            integral = Tr * (2.5 * tau ** 4 - 3 * tau ** 5 + tau ** 6)
            return w * integral, w * sig, w * dsig
        phi_up = w * Tr * 0.5
        if t < self.t_ramp_down:                        # cruise
            return phi_up + w * (t - self.t_start - Tr), w, 0.0
        phi_cruise = phi_up + w * self.T_cruise
        if t < self.t_stop:                             # ramp down (mirror of ramp up)
            tau = (t - self.t_ramp_down) / Tr
            sig = 1 - (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
            dsig = -(30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / Tr
            integral = Tr * (tau - (2.5 * tau ** 4 - 3 * tau ** 5 + tau ** 6))
            return phi_cruise + w * integral, w * sig, w * dsig
        return phi_cruise + w * Tr * 0.5, 0.0, 0.0

    def at(self, t):
        phi, dphi, ddphi = self._phase(t)
        A = self.A
        f = (A * math.sin(phi), 0.5 * A * math.sin(2 * phi))
        df = (A * math.cos(phi), A * math.cos(2 * phi))
        ddf = (-A * math.sin(phi), -2 * A * math.sin(2 * phi))
        p = (self.home[0] + f[0], self.home[1] + f[1], ALT)
        v = (df[0] * dphi, df[1] * dphi, 0.0)
        a = (ddf[0] * dphi ** 2 + df[0] * ddphi, ddf[1] * dphi ** 2 + df[1] * ddphi, 0.0)
        return p, v, a


class TrackingSquare(Node):
    def __init__(self):
        super().__init__('tracking_square')
        self.controller = self.declare_parameter('controller', 'lqr').value
        self.reference = self.declare_parameter('reference', 'smooth').value
        self.trajectory = self.declare_parameter('trajectory', 'square').value
        self.laps = int(self.declare_parameter('laps', 2).value)
        self.tag = self.declare_parameter('tag', '').value
        tag = self.tag
        q_p = float(self.declare_parameter('q_p', LQR_WEIGHTS['xy'][0]).value)
        q_v = float(self.declare_parameter('q_v', LQR_WEIGHTS['xy'][1]).value)
        q_a = float(self.declare_parameter('q_a', 0.1).value)
        r = float(self.declare_parameter('r', LQR_WEIGHTS['xy'][2]).value)
        tau = float(self.declare_parameter('tau', 0.2).value)
        delay = float(self.declare_parameter('delay', 0.08).value)
        mpc_q_p = float(self.declare_parameter('mpc_q_p', 50.0).value)
        mpc_q_v = float(self.declare_parameter('mpc_q_v', 5.0).value)
        mpc_r_du = float(self.declare_parameter('mpc_r_du', 2.0).value)
        if self.trajectory not in ('square', 'figure8'):
            raise ValueError('trajectory must be square|figure8')
        if self.controller not in ('px4', 'pd', 'lqr', 'lqr_lag', 'mpc') or \
                self.reference not in ('linear', 'smooth'):
            raise ValueError('controller must be px4|pd|lqr|lqr_lag|mpc, '
                             'reference must be linear|smooth')

        # xy gains are (K_p, K_v) or (K_p, K_v, K_a); z always uses a 2-state LQR/PD
        if self.controller in ('pd', 'px4'):            # px4: gains unused (PX4 controls)
            self.gains = dict(PD_GAINS)
        elif self.controller == 'lqr':
            self.gains = {'xy': lqr_gains(q_p, q_v, r), 'z': lqr_gains(*LQR_WEIGHTS['z'])}
        elif self.controller == 'lqr_lag':
            self.gains = {'xy': lqr_lag_gains(q_p, q_v, q_a, r, tau, 1.0 / RATE_HZ),
                          'z': lqr_gains(*LQR_WEIGHTS['z'])}
        else:   # mpc for x and y, LQR for height
            self.gains = {'xy': (mpc_q_p, mpc_q_v), 'z': lqr_gains(*LQR_WEIGHTS['z'])}
            self.mpc = [AxisMPC(dt=0.05, horizon=30, tau=tau, q_p=mpc_q_p, q_v=mpc_q_v,
                                r=1.0, r_du=mpc_r_du, u_max=A_MAX_XY) for _ in range(2)]
            self.tau = tau
            self.n_delay = round(delay * RATE_HZ)
            self.sent = deque([(0.0, 0.0)] * self.n_delay, maxlen=max(self.n_delay, 1))
            self.a_est = [0.0, 0.0]          # model-based acceleration estimate
            self.u_last = [0.0, 0.0]
            self.solve_ms = 0.0

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
        self.prev = None
        self.jumps = 0
        self.state = 'INIT'
        self.tick = 0
        self.done = False
        self.climb_t0 = 0.0
        self.t0 = self.now()

        log_path = os.path.expanduser(
            f'~/{self.trajectory}_{self.controller}_{self.reference}{tag}.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.log = csv.writer(self.log_file)
        self.log.writerow(['t', 'state', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                           'ref_x', 'ref_y', 'ref_z', 'ax', 'ay', 'az',
                           'xy_reset', 'z_reset', 'px4_t'])

        self.create_timer(1.0 / RATE_HZ, self.step)
        gains_xy = ' '.join(f'{k:.2f}' for k in self.gains['xy'])
        kp_z, kd_z = self.gains['z']
        self.get_logger().info(
            f'controller={self.controller} reference={self.reference} | '
            f'K_xy=[{gains_xy}] Kp_z={kp_z:.2f} Kd_z={kd_z:.2f} | log: {log_path}')

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

    def publish_setpoint(self, position=None, acceleration=None, velocity=None):
        msg = TrajectorySetpoint()
        msg.position = [float(v) for v in position] if position else [NAN] * 3
        msg.velocity = [float(v) for v in velocity] if velocity else [NAN] * 3
        msg.acceleration = [float(v) for v in acceleration] if acceleration else [NAN] * 3
        msg.yaw = 0.0
        msg.timestamp = self.stamp_us()
        self.setpoint_pub.publish(msg)

    # ---------------- the controller ----------------
    def control(self, p_ref, v_ref, a_ref):
        p = (self.pos.x, self.pos.y, self.pos.z)
        v = (self.pos.vx, self.pos.vy, self.pos.vz)
        a_meas = (self.pos.ax, self.pos.ay, self.pos.az)
        a = []
        for i in range(3):
            k = self.gains['z' if i == 2 else 'xy']
            u = a_ref[i] + k[0] * (p_ref[i] - p[i]) + k[1] * (v_ref[i] - v[i])
            if len(k) == 3:                       # lqr_lag: also use the acceleration error
                u += k[2] * (a_ref[i] - a_meas[i])
            a.append(u)
        a_xy = math.hypot(a[0], a[1])
        if a_xy > A_MAX_XY:
            a[0], a[1] = a[0] * A_MAX_XY / a_xy, a[1] * A_MAX_XY / a_xy
        a[2] = clamp(a[2], A_MAX_Z)
        return tuple(a)

    def control_mpc(self, ts, p_ref, v_ref, a_ref):
        """MPC for x and y (looks ahead along the reference), LQR for z."""
        import time
        t_start = time.perf_counter()
        pending = list(self.sent) if self.n_delay else []
        t_act = ts + self.n_delay / RATE_HZ      # when the next command starts to act
        m = self.mpc[0]
        future = [self.ref.at(t_act + j * m.dt) for j in range(m.N + 1)]
        meas = ((self.pos.x, self.pos.vx), (self.pos.y, self.pos.vy))
        u = [0.0, 0.0]
        for k in range(2):
            x0 = predict_through_delay([meas[k][0], meas[k][1], self.a_est[k]],
                                       [c[k] for c in pending], 1.0 / RATE_HZ, self.tau)
            p_f = [future[j + 1][0][k] for j in range(m.N)]
            v_f = [future[j + 1][1][k] for j in range(m.N)]
            a_f = [future[j][2][k] for j in range(m.N)]
            u[k], _ = self.mpc[k].solve(x0, p_f, v_f, a_f, self.u_last[k])
        a_xy = math.hypot(u[0], u[1])
        if a_xy > A_MAX_XY:
            u = [u[0] * A_MAX_XY / a_xy, u[1] * A_MAX_XY / a_xy]
        # bookkeeping: the command that starts acting now updates the acceleration estimate
        acting = self.sent[0] if self.n_delay else u
        for k in range(2):
            self.a_est[k] += (acting[k] - self.a_est[k]) / (RATE_HZ * self.tau)
        if self.n_delay:
            self.sent.append((u[0], u[1]))
        self.u_last = u
        kp, kd = self.gains['z']
        az = clamp(a_ref[2] + kp * (p_ref[2] - self.pos.z) + kd * (v_ref[2] - self.pos.vz),
                   A_MAX_Z)
        self.solve_ms = max(self.solve_ms, (time.perf_counter() - t_start) * 1000)
        return (u[0], u[1], az)

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
                    self.climb_t0 = t
                    self.state = 'CLIMB'

            if self.state == 'CLIMB' and t - self.climb_t0 > 30.0:
                self.get_logger().warn('Did not reach start point within 30 s. Landing.')
                self.write_result('NOT valid (no start)')
                self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.state = 'LAND'
            if self.state == 'CLIMB':
                dist = math.dist((self.pos.x, self.pos.y, self.pos.z), ref_p)
                if dist < 0.2 and math.hypot(self.pos.vx, self.pos.vy) < 0.2:
                    if self.trajectory == 'figure8':
                        self.ref = FigureEightReference(self.home, laps=self.laps)
                    else:
                        self.ref = SquareReference(self.home, self.reference == 'smooth')
                    self.square_t0 = t
                    self.last_reset = self.pos.xy_reset_counter
                    self.state = 'SQUARE'
                    self.get_logger().info(
                        f'At start. Flying {self.trajectory} with {self.controller.upper()}.')

        elif self.state == 'SQUARE':
            if not self.is_offboard():
                self.get_logger().warn('Left OFFBOARD mode (pilot/failsafe). Stopping.')
                self.done = True
                return
            ts = t - self.square_t0
            px4_t = self.pos.timestamp * 1e-6          # PX4 (simulation) time [s]
            if self.prev is not None:            # position must match velocity * time
                pt, px, py, pvx, pvy, ppx4 = self.prev
                dt = t - pt
                jump = math.hypot(self.pos.x - (px + pvx * dt), self.pos.y - (py + pvy * dt))
                if jump > 0.1:
                    self.jumps += 1
                    self.get_logger().warn(
                        f'Position jumped {jump:.2f} m at {ts:.1f} s. Node time advanced '
                        f'{dt * 1000:.0f} ms, PX4 simulation time advanced '
                        f'{(px4_t - ppx4) * 1000:.0f} ms -> this run is not valid')
            self.prev = (t, self.pos.x, self.pos.y, self.pos.vx, self.pos.vy, px4_t)
            if self.pos.xy_reset_counter != self.last_reset:
                self.get_logger().warn(
                    'PX4 estimator reset its position during the square -> this run is not valid')
                self.last_reset = self.pos.xy_reset_counter
            ref_p, ref_v, ref_a = self.ref.at(ts)
            if self.controller == 'px4':
                # baseline: PX4's own position controller follows the SAME reference,
                # with velocity and acceleration as feed-forward
                acc = ref_a
                self.publish_mode(acceleration=False)
                self.publish_setpoint(position=ref_p, velocity=ref_v, acceleration=ref_a)
            else:
                if self.controller == 'mpc':
                    acc = self.control_mpc(ts, ref_p, ref_v, ref_a)
                else:
                    acc = self.control(ref_p, ref_v, ref_a)
                self.publish_mode(acceleration=True)
                self.publish_setpoint(acceleration=acc)
            if ts > self.ref.end_time:
                verdict = 'VALID run' if self.jumps == 0 else f'NOT valid ({self.jumps} glitches)'
                self.write_result(verdict)
                self.get_logger().info(f'{self.trajectory} complete ({verdict}). Landing.')
                if self.controller == 'mpc':
                    self.get_logger().info(f'Slowest MPC step: {self.solve_ms:.1f} ms '
                                           f'(must stay well below {1000 / RATE_HZ:.0f} ms)')
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
                           f'{acc[0]:.3f}', f'{acc[1]:.3f}', f'{acc[2]:.3f}',
                           self.pos.xy_reset_counter, self.pos.z_reset_counter,
                           f'{self.pos.timestamp * 1e-6:.4f}'])

    def write_result(self, verdict):
        with open(os.path.expanduser('~/runs.txt'), 'a') as f:
            f.write(f'{self.trajectory} {self.controller}{self.tag}: {verdict}\n')

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
