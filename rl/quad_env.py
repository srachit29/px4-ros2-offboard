"""
quad_env.py - fast horizontal quadrotor model for RL training (numpy only).

Plant (per horizontal axis), identified from PX4 SITL flight logs (tools/fit_lag.py):
    a' = (u(t - delay) + d - a) / tau      u: commanded acceleration, d: wind
    v' = a,  p' = v
Domain randomisation per episode: tau 0.15-0.30 s, delay 0-0.06 s, random reference
(figure-eight, circle, smooth random curve, hover), random wind (steady + gust).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'px4_offboard'))
from px4_offboard.rl_common import (DT, OBS_DIM, ACT_DIM, U_MAX, build_obs,  # noqa: E402
                                    update_integral)

EP_TIME = 15.0


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


class RandomReference:
    """Returns (p, v, a) as 2D arrays at time t. Starts slowly (ramp) like the real tests."""

    def __init__(self, rng):
        self.kind = rng.choice(['figure8', 'circle', 'sines', 'hover'], p=[0.4, 0.2, 0.3, 0.1])
        self.A = rng.uniform(1.5, 3.0)
        self.w = 2 * math.pi / rng.uniform(10.0, 16.0)
        self.rot = rng.uniform(0, 2 * math.pi)
        self.ramp = rng.uniform(1.0, 3.0)
        self.freqs = rng.uniform(0.05, 0.2, size=(2, 3))
        self.amps = rng.uniform(0.2, 0.8, size=(2, 3))
        self.phases = rng.uniform(0, 2 * math.pi, size=(2, 3))

    def _pos(self, t):
        # progress with a smooth start: speed ramps 0 -> 1 with a minimum-jerk profile
        if t <= 0.0:
            s = 0.0
        elif t < self.ramp:
            x = t / self.ramp
            s = self.ramp * (2.5 * x ** 4 - 3 * x ** 5 + x ** 6)
        else:
            s = t - self.ramp / 2
        if self.kind == 'figure8':
            phi = self.w * s
            q = np.array([self.A * math.sin(phi), 0.5 * self.A * math.sin(2 * phi)])
        elif self.kind == 'circle':
            phi = self.w * s
            q = np.array([self.A * math.sin(phi), self.A * (1 - math.cos(phi))])
        elif self.kind == 'sines':
            q = np.array([np.sum(self.amps[i] * (np.sin(2 * math.pi * self.freqs[i] * s
                                                         + self.phases[i])
                                                  - np.sin(self.phases[i]))) for i in range(2)])
        else:
            q = np.zeros(2)
        c, sn = math.cos(self.rot), math.sin(self.rot)
        return np.array([c * q[0] - sn * q[1], sn * q[0] + c * q[1]])

    def at(self, t, h=1e-3):
        p = self._pos(t)
        v = (self._pos(t + h) - self._pos(t - h)) / (2 * h)
        a = (self._pos(t + h) - 2 * p + self._pos(t - h)) / h ** 2
        return p, v, a


class QuadTrackCore:
    """Gym-free core so it can be tested without gymnasium installed."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        r = self.rng
        self.tau = r.uniform(0.15, 0.30)
        self.n_delay = int(r.integers(0, 4))
        self.ref = RandomReference(r)
        wind_on = r.random() < 0.6
        self.wind_dir = r.uniform(0, 2 * math.pi)
        self.wind_steady = r.uniform(0.0, 1.3) if wind_on else 0.0
        self.gust_amp = r.uniform(0.0, 1.3) if wind_on else 0.0
        self.gust_f = r.uniform(0.2, 0.8)
        self.wind_start = r.uniform(0.0, 5.0)
        self.t = 0.0
        self.p = self.ref.at(0.0)[0] + r.normal(0, 0.05, 2)
        self.v = np.zeros(2)
        self.a = np.zeros(2)
        self.queue = [np.zeros(2)] * self.n_delay
        self.e_int = np.zeros(2)
        self.u_last = np.zeros(2)
        return self._obs()

    def wind(self, t):
        if t < self.wind_start:
            return np.zeros(2)
        g = self.gust_amp * math.sin(2 * math.pi * self.gust_f * (t - self.wind_start))
        d = np.array([math.cos(self.wind_dir), math.sin(self.wind_dir)])
        n = np.array([-d[1], d[0]])
        return self.wind_steady * d + g * n

    def _obs(self):
        noise = self.rng.normal(0, [0.01, 0.01])
        return build_obs(self.p + noise, self.v + self.rng.normal(0, 0.02, 2), self.ref.at,
                         self.t, self.e_int, self.u_last)

    def step(self, action):
        u = U_MAX * np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        n = np.linalg.norm(u)
        if n > U_MAX:
            u = u * U_MAX / n
        self.queue.append(u)
        u_act = self.queue.pop(0)
        self.a += (u_act + self.wind(self.t) - self.a) * DT / self.tau
        self.v += self.a * DT
        self.p += self.v * DT
        self.t += DT
        p_ref, v_ref, _ = self.ref.at(self.t)
        self.e_int = update_integral(self.e_int, self.p, p_ref)
        e_p = self.p - p_ref
        e_v = self.v - v_ref
        du = u - self.u_last
        self.u_last = u
        reward = -(4.0 * e_p @ e_p + 0.3 * e_v @ e_v + 0.005 * u @ u + 0.02 * du @ du)
        terminated = bool(np.linalg.norm(e_p) > 3.0)
        if terminated:
            reward -= 50.0
        truncated = self.t >= EP_TIME
        info = {'err': float(np.linalg.norm(e_p))}
        return self._obs(), float(reward), terminated, truncated, info


try:
    import gymnasium as gym

    class QuadTrackEnv(gym.Env):
        metadata = {'render_modes': []}

        def __init__(self, seed=None):
            self.core = QuadTrackCore(seed)
            self.observation_space = gym.spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

        def reset(self, seed=None, options=None):
            if seed is not None:
                self.core.rng = np.random.default_rng(seed)
            return self.core.reset(), {}

        def step(self, action):
            return self.core.step(action)
except ImportError:      # allows testing the core without gymnasium
    pass
