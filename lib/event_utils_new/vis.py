import numpy as np
from dv import AedatFile
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

path_to_esot500 = '/root/autodl-tmp/ESOT500'

ae_file = path_to_esot500 + '/aedat4/ball5.aedat4'
with AedatFile(ae_file) as f:
    print('Processing:',ae_file)
    events = np.hstack([packet for packet in f['events'].numpy()])
    events['timestamp'] = events['timestamp'] -events['timestamp'][0]
print(len(events))

# timestamps = events[:]['timestamp'] / 1e6
# print(timestamps)
# index40 = np.searchsorted(timestamps,0.307735)
# index42 = np.searchsorted(timestamps,0.324400)
# print(index40,index42)
# mid_length = index42 - index40
# events = events[index40:index42]
events = events[0:len(events)//2]

xs = events[:]['x']
ys = events[:]['y']
ts = events[:]['timestamp'] / 1e6
ps = events[:]['polarity']

xs = 346 - xs
ys = 260 - ys

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
skip = 5  
marker ='.' 
colors = ['r' if p>0 else ('#00DAFF') for p in ps] 

x_scale=1
y_scale=4
z_scale=1
scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0
def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)
ax.get_proj=short_proj

xs_range=max(xs)-min(xs)
ts_range=max(ts)-min(ts)
ys_range=max(ys)-min(ys)
print(xs_range,max(xs),min(xs))
print(ts_range,max(ts),min(ts))
print(ys_range,max(ys),min(ys))

ax.scatter(xs[::skip]-min(xs)-xs_range/2, 
           ts[::skip]-min(ts)-ts_range/2, 
           ys[::skip]-min(ys)-ys_range/2, 
           zdir='z', c=colors[::skip], facecolors=colors[::skip],
        s=1, marker=marker, linewidths=0, alpha=1.0)


elev = 30
azim = 110
ax.view_init(elev=elev, azim=azim)


ax.set_axis_off()
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# ax.set_xlabel('x') 
ax.set_ylabel('t') 
# ax.set_zlabel('z') 


# ax.set_xlim(-xs_range/2, xs_range/2)
# ax.set_ylim(-ts_range/2, ts_range/2)
# ax.set_zlim(-ys_range/2, ys_range/2)

# plt.show()

plt.savefig('scatter_bottle.png',dpi=600)
