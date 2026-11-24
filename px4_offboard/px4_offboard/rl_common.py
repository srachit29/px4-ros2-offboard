"""
rl_common.py - shared by RL training (rl/train.py) and the ROS 2 node (controller:=rl).

Keeping the observation in ONE place guarantees that the policy sees exactly the same
inputs in training and on the drone.

Observation (16 values, all scaled to roughly [-1, 1]):
    e_p      position error p - p_ref                     (2)   / 1 m
    e_v      velocity error v - v_ref                     (2)   / 1 m/s
    a_ref    planned acceleration of the reference        (2)   / 3 m/s^2
    preview  p_ref(t + T) - p for T = 0.3, 0.6, 1.0 s     (6)   / 3 m
    e_int    integral of the position error (clipped)     (2)   / 1 m s
    u_last   previous command                             (2)   / 3 m/s^2

Action: 2 values in [-1, 1] -> horizontal acceleration command = 3 m/s^2 * action.
"""
import numpy as np

OBS_DIM = 16
ACT_DIM = 2
U_MAX = 3.0
PREVIEW = (0.3, 0.6, 1.0)
E_INT_MAX = 1.0
DT = 0.02          # control period (50 Hz), same as the node


def build_obs(p, v, ref_at, t, e_int, u_last):
    """p, v: measured horizontal position/velocity (2,); ref_at(t) -> (p_ref, v_ref, a_ref)."""
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p_ref, v_ref, a_ref = ref_at(t)
    obs = [(p - np.asarray(p_ref[:2])) / 1.0,
           (v - np.asarray(v_ref[:2])) / 1.0,
           np.asarray(a_ref[:2]) / U_MAX]
    for T in PREVIEW:
        obs.append((np.asarray(ref_at(t + T)[0][:2]) - p) / 3.0)
    obs.append(np.asarray(e_int) / E_INT_MAX)
    obs.append(np.asarray(u_last) / U_MAX)
    return np.clip(np.concatenate(obs), -5.0, 5.0).astype(np.float32)


def update_integral(e_int, p, p_ref):
    """Integral of the position error, clipped (anti-windup)."""
    e = np.asarray(p, dtype=np.float64) - np.asarray(p_ref[:2])
    return np.clip(np.asarray(e_int) + e * DT, -E_INT_MAX, E_INT_MAX)


class MLPPolicy:
    """Numpy forward pass of the trained PPO actor (tanh MLP), no PyTorch needed."""

    def __init__(self, path):
        w = np.load(path)
        self.layers = []
        i = 0
        while f'W{i}' in w:
            self.layers.append((w[f'W{i}'], w[f'b{i}']))
            i += 1

    def act(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return U_MAX * np.clip(x, -1.0, 1.0)
