"""Compare two flights of the square: PX4 position control vs. your controller.

usage: python3 compare_runs.py baseline.csv mine.csv out.png [label_for_mine]
"""
import csv, math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQ = [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)]


def load(path):
    rows = list(csv.DictReader(open(path)))
    sq = [r for r in rows if r['state'] == 'SQUARE']
    t0 = float(sq[0]['t'])
    hx, hy = float(rows[0]['x']), float(rows[0]['y'])
    return dict(t=[float(r['t']) - t0 for r in sq],
                n=[float(r['x']) - hx for r in sq], e=[float(r['y']) - hy for r in sq],
                h=[-float(r['z']) for r in sq])


def seg_dist(px, py, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    u = max(0, min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - a[0] - u * dx, py - a[1] - u * dy)


def metrics(d):
    path = [min(seg_dist(n, e, SQ[k], SQ[k + 1]) for k in range(4)) for n, e in zip(d['n'], d['e'])]
    herr = [abs(h - 5.0) for h in d['h']]
    rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
    return dict(path_rms=rms(path), path_max=max(path), h_rms=rms(herr), h_max=max(herr),
                duration=d['t'][-1] - d['t'][0])


def main():
    base, mine = load(sys.argv[1]), load(sys.argv[2])
    label = sys.argv[4] if len(sys.argv) > 4 else 'Your controller'
    mb, mm = metrics(base), metrics(mine)
    print(f"{'':22s}{'PX4 position':>14s}{label:>18s}")
    for k, name in [('path_rms', 'path error RMS [m]'), ('path_max', 'path error max [m]'),
                    ('h_rms', 'height error RMS [m]'), ('h_max', 'height error max [m]'),
                    ('duration', 'square time [s]')]:
        print(f"{name:22s}{mb[k]:14.3f}{mm[k]:18.3f}")

    c_base, c_mine = '#9a9a9a', '#5B3A99'
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.plot([p[1] for p in SQ], [p[0] for p in SQ], ':', color='k', lw=1, label='Planned square')
    a1.plot(base['e'], base['n'], color=c_base, lw=2, label='PX4 position control')
    a1.plot(mine['e'], mine['n'], color=c_mine, lw=2, label=label)
    a1.set_aspect('equal'); a1.grid(alpha=.3); a1.set_xlabel('East [m]'); a1.set_ylabel('North [m]')
    a1.set_title('Top view'); a1.legend(loc='center', fontsize=9)
    a2.plot(base['t'], base['h'], color=c_base, lw=2, label='PX4 position control')
    a2.plot(mine['t'], mine['h'], color=c_mine, lw=2, label=label)
    a2.axhline(5, ls=':', color='k', lw=1)
    a2.set_xlabel('Time since square start [s]'); a2.set_ylabel('Height [m]'); a2.grid(alpha=.3)
    a2.set_title('Height during the square'); a2.legend(fontsize=9)
    fig.suptitle('Square tracking: PX4 position control vs. ' + label, fontsize=12)
    fig.tight_layout(); fig.savefig(sys.argv[3], dpi=150)


if __name__ == '__main__':
    main()
