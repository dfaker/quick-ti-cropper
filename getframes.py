import os
import sys

files = sys.argv[1:]
delay_seconds = 60

for p in files:
    basename = os.path.basename(p)

    os.path.exists('frames') or os.mkdir('frames')

    if os.path.exists('frames\\' + p + '_000001.png'):
        continue

    cmd = 'ffmpeg -i "{}" -vsync vfr -vf select=eq(pict_type\\,I)*(isnan(prev_selected_t)+gte(t-prev_selected_t\\,{})) -f image2 "frames\\{}_%06d.png"'.format(
        p, delay_seconds, basename)
    print(cmd)
    os.system(cmd)
