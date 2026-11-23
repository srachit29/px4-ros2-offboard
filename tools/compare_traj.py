"""Compare flights on any trajectory (square or figure-eight) using the logged reference.

usage: python3 compare_traj.py out.png "label1=log1.csv" "label2=log2.csv" ...

Metrics (only while the reference is active, state SQUARE in the log):
  tracking error  distance to where the drone SHOULD be at that moment
  path error      distance to the nearest point of the reference path (ignores timing)
  height error    |height - 5 m|
"""
import csv
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#9a9a9a', '#e07b39', '#2a9d8f', '#5B3A99', '#c0392b']


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r['state'] == 'SQUARE']
    t0 = float(rows[0]['t'])
    hx, hy = float(rows[0]['ref_x']), float(rows[0]['ref_y'])
    f = lambda key, off=0.0: [float(r[key]) - off for r in rows]
    return dict(t=[float(r['t']) - t0 for r in rows],
                n=f('x', hx), e=f('y', hy), rn=f('ref_x', hx), re=f('ref_y', hy),
                h=[-float(r['z']) for r in rows])


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    u = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - ax - u * dx, py - ay - u * dy)


def metrics(d):
    rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
    trk = [math.hypot(n - rn, e - re) for n, e, rn, re in zip(d['n'], d['e'], d['rn'], d['re'])]
    # reference path as a polyline, thinned to ~2 cm spacing for speed
    pts = [(d['rn'][0], d['re'][0])]
    for p in zip(d['rn'], d['re']):
        if math.dist(p, pts[-1]) > 0.02:
            pts.append(p)
    path = [min(seg_dist(n, e, *pts[i], *pts[i + 1]) for i in range(len(pts) - 1))
            for n, e in zip(d['n'], d['e'])]
    herr = [abs(h - 5.0) for h in d['h']]
    return dict(trk_rms=rms(trk), trk_max=max(trk), path_rms=rms(path), path_max=max(path),
                h_rms=rms(herr), h_max=max(herr), trk=trk)


def main():
    out = sys.argv[1]
    runs = [(a.split('=', 1)[0], load(a.split('=', 1)[1])) for a in sys.argv[2:]]
    print('| Controller | Tracking error RMS | Tracking error max | Path error RMS '
          '| Path error max | Height error RMS |')
    print('|---|---|---|---|---|---|')
    res = []
    for name, d in runs:
        m = metrics(d)
        res.append(m)
        print(f"| {name} | {m['trk_rms']*100:.1f} cm | {m['trk_max']*100:.1f} cm "
              f"| {m['path_rms']*100:.1f} cm | {m['path_max']*100:.1f} cm | {m['h_rms']*100:.1f} cm |")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8),
                                     gridspec_kw={'width_ratios': [1.1, 1.3, 1.3]})
    ref = runs[0][1]
    a1.plot(ref['re'], ref['rn'], ':', color='k', lw=1.2, label='Reference')
    for (name, d), m, c in zip(runs, res, COLORS):
        a1.plot(d['e'], d['n'], color=c, lw=1.6, label=name)
        a2.plot(d['t'], [x * 100 for x in m['trk']], color=c, lw=1.4, label=name)
        a3.plot(d['t'], d['h'], color=c, lw=1.4, label=name)
    a1.set_aspect('equal'); a1.grid(alpha=.3); a1.set_title('Top view')
    a1.set_xlabel('East [m]'); a1.set_ylabel('North [m]'); a1.legend(fontsize=8, loc='lower right')
    a2.grid(alpha=.3); a2.set_title('Tracking error (distance to moving target)')
    a2.set_xlabel('Time [s]'); a2.set_ylabel('Error [cm]'); a2.legend(fontsize=8)
    a3.axhline(5, ls=':', color='k', lw=1); a3.grid(alpha=.3); a3.set_title('Height')
    a3.set_xlabel('Time [s]'); a3.set_ylabel('Height [m]')
    fig.suptitle('Trajectory tracking in PX4 SITL (Gazebo x500)')
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
