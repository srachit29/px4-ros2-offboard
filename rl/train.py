"""
train.py - train a PPO tracking controller and export it for the ROS 2 node.

usage (inside the RL virtual environment):
    python rl/train.py                    # 2 M steps, ~20-40 min on 16 cores
    python rl/train.py --steps 300000     # quick test

Output:
    rl/ppo_quad.zip       full Stable-Baselines3 model (to continue training)
    rl/policy.npz         actor weights for the ROS node (numpy only)
    rl/train_log.csv      mean episode reward and tracking error during training
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quad_env import QuadTrackEnv  # noqa: E402


class Progress(BaseCallback):
    """Prints and logs mean reward / tracking error per rollout."""

    def __init__(self, path):
        super().__init__()
        self.f = open(path, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(['steps', 'mean_reward_per_step', 'mean_err_cm'])
        self.rew, self.err, self.t0 = [], [], time.time()

    def _on_step(self):
        self.rew.extend(self.training_env.get_original_reward())
        self.err.extend(i['err'] for i in self.locals['infos'])
        return True

    def _on_rollout_end(self):
        r, e = float(np.mean(self.rew)), float(np.mean(self.err)) * 100
        self.w.writerow([self.num_timesteps, f'{r:.4f}', f'{e:.2f}'])
        self.f.flush()
        print(f'{self.num_timesteps:>9d} steps | reward/step {r:8.3f} | '
              f'mean error {e:6.1f} cm | {time.time() - self.t0:5.0f} s', flush=True)
        self.rew, self.err = [], []


def export(model, path):
    """Save actor weights (Linear layers of policy_net + action_net) as numpy arrays."""
    layers = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, 'weight')]
    layers.append(model.policy.action_net)
    out = {}
    for i, layer in enumerate(layers):
        out[f'W{i}'] = layer.weight.detach().cpu().numpy()
        out[f'b{i}'] = layer.bias.detach().cpu().numpy()
    np.savez(path, **out)
    print('exported', path, [out[f'W{i}'].shape for i in range(len(layers))])


def make_env(i):
    return lambda: QuadTrackEnv(seed=1000 + i)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2_000_000)
    ap.add_argument('--envs', type=int, default=8)
    args = ap.parse_args()

    env = SubprocVecEnv([make_env(i) for i in range(args.envs)])
    # scale rewards only (observations are already scaled in rl_common.build_obs, so the
    # exported policy needs no normalisation statistics on the drone)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO('MlpPolicy', env, n_steps=1024, batch_size=1024, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.0, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.0),
                device='cpu', verbose=0, seed=0)
    model.learn(total_timesteps=args.steps, callback=Progress(os.path.join(HERE, 'train_log.csv')))
    model.save(os.path.join(HERE, 'ppo_quad'))
    export(model, os.path.join(HERE, 'policy.npz'))
