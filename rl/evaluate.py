"""
evaluate.py - compare the trained policy with LQI in the training simulator (numpy only).

usage: python rl/evaluate.py [rl/policy.npz]
Runs 50 random episodes (same seeds for both controllers) and prints mean tracking error.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'px4_offboard'))
from quad_env import QuadTrackCore  # noqa: E402
from px4_offboard.rl_common import MLPPolicy, U_MAX  # noqa: E402

KI, KP, KV = 2.0, 4.3, 3.6     # LQI-like reference gains (for comparison only)


def lqi(env):
    p_ref, v_ref, a_ref = env.ref.at(env.t)
    u = a_ref + KP * (p_ref - env.p) + KV * (v_ref - env.v) - KI * env.e_int
    return u / U_MAX


def run(policy, seeds):
    errs = []
    for s in seeds:
        env = QuadTrackCore(seed=s)
        obs = env.reset()
        e, done = [], False
        while not done:
            act = policy(env, obs)
            obs, _, term, trunc, info = env.step(act)
            e.append(info['err'])
            done = term or trunc
        errs.append(np.sqrt(np.mean(np.square(e))))
    return np.array(errs) * 100


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'policy.npz')
    seeds = range(5000, 5050)
    base = run(lambda env, obs: lqi(env), seeds)
    print(f'LQI : mean RMS error {base.mean():5.1f} cm (median {np.median(base):5.1f})')
    if os.path.exists(path):
        pol = MLPPolicy(path)
        rl = run(lambda env, obs: pol.act(obs) / U_MAX, seeds)
        print(f'RL  : mean RMS error {rl.mean():5.1f} cm (median {np.median(rl):5.1f})')
        print(f'RL better than LQI in {np.sum(rl < base)} of {len(seeds)} episodes')
    else:
        print('no policy found at', path)
