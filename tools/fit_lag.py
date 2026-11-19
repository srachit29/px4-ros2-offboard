"""Estimate how fast the drone follows an acceleration command.

Model per horizontal axis:  a_actual' = (a_cmd(t - delay) - a_actual) / tau,  v' = a_actual
Fits delay and tau by simulating the velocity over short windows and comparing with the log.
usage: python3 fit_lag.py log.csv [log2.csv ...]
"""
import csv, math, sys


def load(f):
    r = [x for x in csv.DictReader(open(f)) if x['state'] == 'SQUARE' and x['ax'] != 'nan']
    return ([float(x['t']) for x in r], [[float(x['ax']), float(x['ay'])] for x in r],
            [[float(x['vx']), float(x['vy'])] for x in r])


def score(t, a, v, d, tau, win=50):
    err = n = 0
    steps = round(d / 0.02)
    for s in range(steps + 1, len(t) - win - 1, 25):
        vs = list(v[s])
        aa = [(v[s + 1][k] - v[s - 1][k]) / (t[s + 1] - t[s - 1]) for k in range(2)]
        for i in range(s, s + win):
            dt = t[i + 1] - t[i]
            for k in range(2):
                aa[k] += (a[i - steps][k] - aa[k]) * dt / tau
                vs[k] += aa[k] * dt
            err += sum((vs[k] - v[i + 1][k]) ** 2 for k in range(2)); n += 1
    return math.sqrt(err / n)


for f in sys.argv[1:]:
    t, a, v = load(f)
    best = min((score(t, a, v, d, tau), d, tau)
               for d in [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2]
               for tau in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6])
    print(f'{f}: delay {best[1]:.2f} s, lag tau {best[2]:.2f} s  (velocity fit error {best[0]:.3f} m/s)')
