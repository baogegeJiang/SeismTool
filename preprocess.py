import obspy
from matplotlib import pyplot as plt
file = '/home/jiangyr/shot_ikey20.segy'
st = obspy.read(file,unpack_trace_headers=True)

keyXG='group_coordinate_x'
keyYG='group_coordinate_y'
keyXS='source_coordinate_x'
keyYS='source_coordinate_y'
plt.figure(figsize=(6,6))
for ST in st:
    plt.plot(ST.stats['segy']['trace_header'][keyXG]-ST.stats['segy']['trace_header'][keyXS],ST.stats['segy']['trace_header'][keyYG]-ST.stats['segy']['trace_header'][keyYS],'.k',markersize=0.5)
plt.savefig('res/log.jpg',dpi=300)