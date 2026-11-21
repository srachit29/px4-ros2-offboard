"""Compare several square flights.

usage: python3 compare_all.py out.png label1=log1.csv label2=log2.csv ...
Prints a metrics table (markdown) and saves a top view + corner zoom + height plot.
"""
import csv, math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQ = [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)]
COLORS = ['#9a9a9a', '#e07b39', '#2a9d8f', '#5B3A99', '#c0392b']


def load(path):
    rows = list(csv.DictReader(open(path)))
    sq = [r for r in rows if r['state'] == 'SQUARE']
    t0 = float(sq[0]['t'])
    hx, hy = float(rows[0]['x']), float(rows[0]['y'])
    trk = [math.hypot(float(r['x']) - float(r['ref_x']), float(r['y']) - float(r['ref_y']))
           for r in sq] if 'ref_x' in sq[0] else None
    return dict(trk=trk, t=[float(r['t']) - t0 for r in sq],
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
    trk = rms(d['trk']) if d['trk'] else float('nan')
    return path_rms_max(path) + (rms(herr), max(herr), trk)


def path_rms_max(path):
    return (math.sqrt(sum(x * x for x in path) / len(path)), max(path))


def main():
    out = sys.argv[1]
    runs = [(a.split('=', 1)[0], load(a.split('=', 1)[1])) for a in sys.argv[2:]]
    print('| Controller | Path error RMS | Path error max | Error to moving target RMS '
          '| Height error RMS | Height error max |')
    print('|---|---|---|---|---|---|')
    for name, d in runs:
        pr, pm, hr, hm, tr = metrics(d)
        print(f'| {name} | {pr*100:.1f} cm | {pm*100:.1f} cm | {tr*100:.1f} cm '
              f'| {hr*100:.1f} cm | {hm*100:.1f} cm |')

    fig = plt.figure(figsize=(14, 4.8))
    a1, a2, a3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
    for ax in (a1, a2):
        ax.plot([p[1] for p in SQ], [p[0] for p in SQ], ':', color='k', lw=1)
    for (name, d), c in zip(runs, COLORS):
        a1.plot(d['e'], d['n'], color=c, lw=1.8, label=name)
        a2.plot(d['e'], d['n'], color=c, lw=2)
        a3.plot(d['t'], d['h'], color=c, lw=1.6, label=name)
    a1.set_aspect('equal'); a1.grid(alpha=.3); a1.set_title('Top view')
    a1.set_xlabel('East [m]'); a1.set_ylabel('North [m]'); a1.legend(loc='center', fontsize=8)
    a2.set_xlim(4.4, 5.9); a2.set_ylim(4.4, 5.9); a2.set_aspect('equal'); a2.grid(alpha=.3)
    a2.set_title('Zoom: corner (5, 5)'); a2.set_xlabel('East [m]')
    a3.axhline(5, ls=':', color='k', lw=1); a3.grid(alpha=.3); a3.set_title('Height')
    a3.set_xlabel('Time since square start [s]'); a3.set_ylabel('Height [m]')
    fig.suptitle('Square tracking: PX4 position control vs. own controllers (PX4 SITL, Gazebo x500)')
    fig.tight_layout(); fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
