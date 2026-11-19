import sys, csv, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open(sys.argv[1])))
t=[float(r['t']) for r in rows]; x=[float(r['x']) for r in rows]; y=[float(r['y']) for r in rows]
h=[-float(r['z']) for r in rows]; st=[r['state'] for r in rows]
hx,hy=x[0],y[0]
sq=[(0,0),(5,0),(5,5),(0,5),(0,0)]
# errors during SQUARE phase (cross-track to segment path)
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.6))
a1.plot([hy+p[1] for p in sq],[hx+p[0] for p in sq],'--',color='#999',lw=1.5,label='Planned square')
fx=[x[i] for i in range(len(rows)) if st[i] in('CLIMB','SQUARE')]
fy=[y[i] for i in range(len(rows)) if st[i] in('CLIMB','SQUARE')]
a1.plot(fy,fx,color='#5B3A99',lw=2,label='Flown path')
for i,p in enumerate(sq[:4]): a1.annotate(f'C{i}',(hy+p[1],hx+p[0]),xytext=(6,6),textcoords='offset points',fontsize=9)
a1.set_xlabel('East [m]'); a1.set_ylabel('North [m]'); a1.set_aspect('equal'); a1.grid(alpha=.3)
a1.set_title('Top view'); a1.legend(loc='center',fontsize=9)
a2.plot(t,h,color='#5B3A99',lw=2,label='Height')
a2.axhline(5,ls='--',color='#999',lw=1.2,label='Target 5 m')
cols={'CLIMB':'#e8f0fb','SQUARE':'#eaf6ec','LAND':'#fbeee8'}
for s,c in cols.items():
    idx=[i for i in range(len(st)) if st[i]==s]
    if idx: a2.axvspan(t[idx[0]],t[idx[-1]],color=c,zorder=0); a2.text((t[idx[0]]+t[idx[-1]])/2,5.6,s.lower(),ha='center',fontsize=9,color='#555')
a2.set_ylim(-0.3,6.2); a2.set_xlabel('Time [s]'); a2.set_ylabel('Height [m]'); a2.grid(alpha=.3); a2.set_title('Height over time'); a2.legend(loc='lower center',fontsize=9)
fig.suptitle('PX4 offboard square mission (ROS 2 Jazzy, Gazebo x500 SITL)',fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[2],dpi=150)
# stats
sq_i=[i for i in range(len(st)) if st[i]=='SQUARE']
hs=[h[i] for i in sq_i]
print('duration',t[-1]-t[0],'square alt mean',sum(hs)/len(hs),'min',min(hs),'max',max(hs))
# max deviation from planned square perimeter
def segd(px,py,ax,ay,bx,by):
    dx,dy=bx-ax,by-ay; L=dx*dx+dy*dy; u=max(0,min(1,((px-ax)*dx+(py-ay)*dy)/L)); return math.hypot(px-ax-u*dx,py-ay-u*dy)
d=[min(segd(x[i]-hx,y[i]-hy,*sq[k],*sq[k+1]) for k in range(4)) for i in sq_i]
print('max dev',max(d),'rms',math.sqrt(sum(v*v for v in d)/len(d)))
