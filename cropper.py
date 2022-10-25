import cv2
import os
import numpy as np
import random
import sys
import os
import math
import mimetypes
import subprocess as sp
import json
from threading import Thread
from queue import Queue
import glob
import time
from collections import deque

print("""
Pass a folder of images or videos, or single files  as a prameter:

cropper.py [OptionalArgs] Z:\\somefolder\\somewhere
cropper.py [OptionalArgs] Z:\\somefolder\\somewhere\\test.png
cropper.py [OptionalArgs] Z:\\somefolder\\somewhere\\test.mp4

A and D to go back and forwards in frame history
Right Click or Space for next image.
Mousewheel for resizing the box, Shift tot resize faster
Left Click to capture the hilighted square
Left Click and Drag to evenly spread a line of crops.
Ctrl and Mousewheel Change the overlap of crops when dragging.
B to skip to the next video during video frame cropping.
R to rotate the image clockwise.

Optional Args:

General options:
    SHUFFLE - Shuffle Images.
    ONECLICK - Jump to next image on every click.
    LARGESTFIRST - Sort files by size, largest to smallest.
    MAXCROP - Initialize on each image with the largest possible crop size.
    DARKENBG - Darken the unselected areas of the image in preview.
    WHOLEFRAME - don't crop, take the whole frame
    SLIDESHOW_TIMEOUT - timeout to skip to next image automatically
    SKIPTXT - Do not create caption .txt files
    NOSCALE - Do not scale final images down to TARGET_SIZExTARGET_SIZE
    NOIMAGES - Skip display of image files.
    NOVIDEO - Skip display of video frames.
    MAX_FILE_SIZE=N - Do not load files over N MB in size.
    TARGET_SIZE=N - Set the target dimension in pixels (Default 512)
    OUTDIR=DIR - Output to a directory called DIR
    ALLOW_SMALLER_CROP - Allow crops smaller than TARGET_SIZExTARGET_SIZE

Video Options:
    IFRAME_GAP=N - When cropping video frames, wait at least N seconds between Key frames (Default 10).
    VIDEO_START=N - When cropping video frames, Sart this many seconds or HH:MM:SS into the video

Cache Limits:
    FRAME_FWD_CACHE=N - The maximum number of bytes to use for a forward frame cache when cropping video frames (Default 1GB).
    FRAME_HIST_CACHE=N - The maximum number of bytes to use to keep images in history so they can be skipped back to (Default 1GB).

images are saved into .\\outdir\\

""")


def human_readable_to_bytes(size):
    """Given a human-readable byte string (e.g. 2G, 10GB, 30MB, 20KB),
      return the number of bytes.  Will return 0 if the argument has
      unexpected form.
    """
    if (size[-1] == 'B'):
        size = size[:-1]
    if (size.isdigit()):
        bytes = int(size)
    else:
        bytes = size[:-1]
        unit = size[-1]
        if (bytes.isdigit()):
            bytes = int(bytes)
            if (unit == 'G'):
                bytes *= 1073741824
            elif (unit == 'M'):
                bytes *= 1048576
            elif (unit == 'K'):
                bytes *= 1024
            else:
                bytes = 0
        else:
            bytes = 0
    return bytes, size+'B'


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

if len(sys.argv) < 2:
    print('Pass in a source folder as a parameter')
    exit()

files = []
shuffle = False
one_click = False
largest_first = False
max_crop = False
darken_bg = False
seconds_between_iframes = 10
video_start = 0
fullscreen = True
whole_frame_mode = False
outdir = 'outdir'
slideshow_timeout=None
target_size = 512
scale_to_target = True
skip_txt = False
prevent_smaller_than_target = True
skip_smaller_than_target = True
show_images = True
show_video = True
max_size = None
correct_sar = True
crop_ar = 1

history_max_bytes, _ = human_readable_to_bytes('1GB')
frame_fwd_cache_bytes, _ = human_readable_to_bytes('1GB')

httpPrefixes = ['HTTP:', 'HTTPS:', 'WWW.']
skip_seen = False


def generateFileNames():
    global shuffle, one_click, largest_first, max_crop, darken_bg, seconds_between_iframes, video_start, fullscreen
    global whole_frame_mode, outdir, slideshow_timeo, target_size, scale_to_target, skip_txt
    global prevent_smaller_than_target, skip_smaller_than_target, show_images, show_video, max_size
    global skip_seen, history_max_bytes, frame_fwd_cache_bytes

    files = []

    for source_folder in sys.argv[1:]:
        if source_folder.upper() == 'SHUFFLE':
            print('ARG SHUFFLE=True')
            shuffle = True
            continue
        if source_folder.upper() == 'ONECLICK':
            print('ARG ONECLICK=True')
            one_click = True
            continue
        if source_folder.upper() == 'LARGESTFIRST':
            print('ARG LARGESTFIRST=True')
            largest_first = True
            continue
        if source_folder.upper() == 'MAXCROP':
            print('ARG MAXCROP=True')
            max_crop = True
            continue
        if source_folder.upper() == 'NOFULLSCREEN':
            print('ARG NOFULLSCREEN=True')
            fullscreen = False
            continue
        if source_folder.upper() == 'DARKENBG':
            print('ARG DARKENBG=True')
            darken_bg = True
            continue
        if source_folder.upper() == 'WHOLEFRAME':
            print('ARG WHOLEFRAME=True')
            whole_frame_mode = True
            continue
        if source_folder.upper() == 'SKIPTXT':
            print('ARG SKIPTXT=True')
            skip_txt = True
            continue
        if source_folder.upper() == 'NOSCALE':
            print('ARG NOSCALE=True')
            scale_to_target = False
            continue
        if source_folder.upper() == 'NOIMAGES':
            print('ARG NOIMAGES=True')
            show_images = False
            continue
        if source_folder.upper() == 'NOVIDEO':
            print('ARG NOVIDEO=True')
            show_video = False
            continue
        if source_folder.upper() == 'ALLOW_SMALLER_CROP':
            print('ARG ALLOW_SMALLER_CROP=True')
            prevent_smaller_than_target = False
            continue
        if source_folder.upper() == 'SKIP_SEEN':
            print('ARG SKIP_SEEN=True')
            skip_seen = True
            continue

        if 'CROP_AR=' in source_folder.upper():
            try:
                ar_value = source_folder.split('=')[-1]
                if ':' in ar_value:
                    a, b = ar_value.split(':')
                    crop_ar = float(a)/float(b)
                else:
                    crop_ar = float(ar_value)
                print('ARG CROP_AR=', crop_ar)
            except Exception as e:
                print('Setting ARG CROP_AR Failed')
            continue

        if 'FRAME_HIST_CACHE=' in source_folder.upper():
            try:
                history_max_bytes, _ = human_readable_to_bytes(source_folder.split('=')[-1])
                print('ARG FRAME_HIST_CACHE=', sizeof_fmt(history_max_bytes))
            except Exception as e:
                print('Setting ARG FRAME_HIST_CACHE Failed')
            continue

        if 'FRAME_FWD_CACHE=' in source_folder.upper():
            try:
                frame_fwd_cache_bytes,_ = human_readable_to_bytes(source_folder.split('=')[-1])
                print('ARG FRAME_FWD_CACHE=', sizeof_fmt(frame_fwd_cache_bytes))
            except Exception as e:
                print('Setting ARG FRAME_FWD_CACHE Failed')
            continue

        if 'MAX_FILE_SIZE=' in source_folder.upper():
            try:
                max_size = float(source_folder.split('=')[-1])
                print('ARG MAX_FILE_SIZE=', max_size)
            except Exception as e:
                print('Setting ARG MAX_FILE_SIZE Failed')
            continue

        if 'TARGET_SIZE=' in source_folder.upper():
            try:
                target_size = int(source_folder.split('=')[-1])
                print('ARG TARGET_SIZE=', target_size)
            except Exception as e:
                print('Setting ARG TARGET_SIZE Failed')
            continue

        if 'SLIDESHOW_TIMEOUT=' in source_folder.upper():
            try:
                slideshow_timeout = float(source_folder.split('=')[-1])
                print('ARG SLIDESHOW_TIMEOUT=', slideshow_timeout)
            except Exception as e:
                print('Setting ARG SLIDESHOW_TIMEOUT Failed')
            continue

        if 'IFRAME_GAP=' in source_folder.upper():
            try:
                seconds_between_iframes = float(source_folder.split('=')[-1])
                print('ARG IFRAME_GAP=', seconds_between_iframes)
            except Exception as e:
                print('Setting ARG IFRAME_GAP Failed')
            continue

        if 'VIDEO_START=' in source_folder.upper():
            try:
                video_start = source_folder.split('=')[-1]
                print('ARG VIDEO_START=', video_start)
            except Exception as e:
                print('Setting ARG VIDEO_START Failed')
            continue

        if 'OUTDIR=' in source_folder.upper():
            try:
                outdir = source_folder.split('=')[-1].strip('" ')
                print('ARG OUTDIR=', outdir)
            except Exception as e:
                print('Setting ARG OUTDIR Failed')
            continue

        if os.path.isfile(source_folder):
            if shuffle or largest_first:
                files.append(source_folder)
            else:
                yield source_folder
        elif any([source_folder.upper().startswith(x) for x in httpPrefixes]):
            if shuffle or largest_first:
                files.append(source_folder)
            else:
                yield source_folder
        elif os.path.isdir(source_folder):
            for r, dl, fl in os.walk(source_folder):
                print('source_folder', r, len(fl), len(files))
                for f in fl:
                    
                    if shuffle or largest_first:
                        files.append(os.path.join(r, f))
                    else:
                        yield os.path.join(r, f)
        else:
            globFound = False
            for root in glob.glob(source_folder):
                globFound = True
                if os.path.isfile(root):
                    print('source_folder', root, len(files))
                    if shuffle or largest_first:
                        files.append(root)
                    else:
                        yield root
                else:
                    for r, dl, fl in os.walk(root):
                        print('source_folder', r, len(fl), len(files))
                        for f in fl:
                            if shuffle or largest_first:
                                files.append(os.path.join(r, f))
                            else:
                                yield os.path.join(r, f)
            if not globFound:
                print(source_folder, 'is not a valid file, directory or glob')
                exit()

    if shuffle:
        random.shuffle(files)
        for file in files:
            yield file
    elif largest_first:
        files = sorted(files, key=lambda x: os.stat(x).st_size, reverse=True)
        for file in files:
            yield file


lastx, lasty = 0, 0
coord_list = []
dim = target_size
grabNow = False
skipNow = False
draggStart = None
overlap = 1.0


def on_mouse(event, x, y, flags, param):
    global lastx, lasty, dim, grabNow, skipNow, draggStart, coord_list, overlap
    lastx, lasty = (x, y)

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if flags < 0:
                overlap -= 0.1
            else:
                overlap += 0.1
            overlap = round(overlap, 2)
        else:
            factor = 16
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                factor = 64

            if flags > 0:
                dim += factor
            elif flags < 0:
                dim -= factor

        dim = max(16, dim)
    elif event == cv2.EVENT_LBUTTONDOWN:
        draggStart = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = None
        draggStart = None
        grabNow = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        draggStart = None
        skipNow = True



n = 0
font = cv2.FONT_HERSHEY_SIMPLEX
lastsave = None
lastGrabs = []

BUFSIZE = 10**5

popen_params = {"bufsize": BUFSIZE,
                "stdout": sp.PIPE,
                "stderr": sp.DEVNULL}


class ThreadedGenerator(object):

    def __init__(self, iterator,
                 sentinel=object(),
                 queue_maxsize=0,
                 queue_maxsize_bytes=0,
                 daemon=False,
                 Thread=Thread,
                 Queue=Queue):
        self.queue_maxsize_bytes = queue_maxsize_bytes
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon = daemon

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def getQueueSize(self):
        return self._queue.qsize()

    def _run(self):
        try:
            for value in self._iterator:
                while self.queue_maxsize_bytes > 0 and self._queue.qsize() * value[1].nbytes > self.queue_maxsize_bytes:
                    time.sleep(1)
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value

        self._thread.join()


class VideoFrameGenerator:

    def __init__(self, video_path):
        self.video_path = video_path

        procInfo = sp.Popen(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path], **popen_params)
        outs, errs = procInfo.communicate()
        procInfo = json.loads(outs)
        w, h = None, None

        for stream in procInfo['streams']:

            if 'width' in stream and 'height' in stream:
                sar = 1
                if correct_sar:
                    try:
                        a, b = stream['sample_aspect_ratio'].split(':')
                        sar = float(a)/float(b)
                    except Exception as e:
                        pass

                w = stream['width']
                if correct_sar:
                    w = int(stream['width']*sar)

                h = stream['height']

        filter_exp = 'select=(isnan(prev_selected_t)+gte(t-prev_selected_t\\,{}))'.format(seconds_between_iframes),
        if correct_sar:
            'select=(isnan(prev_selected_t)+gte(t-prev_selected_t\\,{})),scale=w=in_w*sar:h=in_h,setsar=1'.format(seconds_between_iframes),

        self.proc = None
        self.nbytes = None
        if w and h:
            cmd = (['ffmpeg', "-discard", "nokey",
                    '-ss', '{}'.format(video_start),
                    '-i', video_path,
                    '-loglevel',  'error',
                    '-f',         'image2pipe',
                    "-an", "-sn", '-dn',
                    "-bufsize",   "20M",
                    '-vf',        'select=(isnan(prev_selected_t)+gte(t-prev_selected_t\\,{})),scale=w=in_w*sar:h=in_h,setsar=1'.format(seconds_between_iframes),
                    "-pix_fmt",   'bgr24',
                    "-vsync",     "vfr",
                    '-vcodec',    'rawvideo', '-'])

            self.proc = sp.Popen(cmd, **popen_params)
            self.nbytes = 3 * w * h
            self.h = h
            self.w = w

    def __iter__(self):
        n = 0
        while self.proc is not None:
            s = self.proc.stdout.read(self.nbytes)
            n += 1
            if len(s) == self.nbytes:
                result = np.frombuffer(s, dtype='uint8')
                result.shape = (self.h, self.w, 3)
                yield str(n)+self.video_path, result
            else:
                break

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.proc.kill()
        self.proc.communicate()
        self.proc.kill()


threadGen = None


def frameGenerator(file_path):
    global threadGen

    if max_size is not None and os.path.exists(file_path) and os.stat(file_path).st_size > max_size*1024*1024:
        pass
    else:
        imo = None
        if show_images:
            imo = cv2.imread(file_path.strip())

        if imo is not None:
            yield 'Image Still', file_path, imo
        elif show_video:
            try:
                with VideoFrameGenerator(file_path) as videoFrameGen:
                    threadGen = ThreadedGenerator(videoFrameGen, queue_maxsize=100, queue_maxsize_bytes=frame_fwd_cache_bytes, daemon=True)
                    for f, imo in threadGen:
                        yield 'Video Frame', f, imo
            except Exception as e:
                pass


def fileGenerator(files):
    for f in files:
        yield f


fngen = generateFileNames()
fgen = fileGenerator(fngen)
fg = None
k = ord('q')
frame_type = None

window_initialised = False

history = deque([], maxlen=100)
historyInd = 0

seen = set()
seen_files_path = os.path.join(outdir, 'seen_files.json')


def get_history_size():
    total = 0
    for _, _, _, _, _, imo, padded_im in history:
        total += (imo.nbytes + padded_im.nbytes)
    return total


try:
    seen = set(json.loads(open(seen_files_path,'r').read()))
except Exception as e:
    print(e)

for fi, base_file in enumerate(fgen):

    if skip_seen and base_file in seen:
        continue

    seen.add(base_file)

    if not window_initialised:
        print('\nStarting\n')
        cv2.namedWindow("imageWindow", cv2.WND_PROP_FULLSCREEN)
        if fullscreen:
            cv2.setWindowProperty("imageWindow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.setMouseCallback('imageWindow', on_mouse)
        window_initialised = True

    if fg is None:
        rect = cv2.getWindowImageRect('imageWindow')
        _, _, w, h = rect
        fg = np.zeros((h, w, 3), np.uint8)

    fg = np.zeros_like(fg)
    text_size = cv2.getTextSize('Loading Next Image from {}'.format(base_file), font, 0.5, 1)[0]
    fg = cv2.putText(fg, 'Loading Next Image from {}'.format(base_file), (fg.shape[1]//2-(text_size[0]//2), fg.shape[0]//2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('imageWindow', fg)

    k = cv2.pollKey()

    for iframe, (frame_type, f, imo) in enumerate(frameGenerator(base_file)):

        skip = False
        norm_path = os.path.normpath(f)
        norm_path = [xp.replace('&', ' ') for xp in norm_path.split(os.sep)[2:-1]]

        rect = cv2.getWindowImageRect('imageWindow')
        _, _, w, h = rect

        old_size = imo.shape[:2]
        oh, ow = old_size

        ratiow = float(w)/float(ow)
        ratioh = float(h)/float(oh)

        ratio = min(min(ratiow, ratioh), 1)

        new_size = tuple([int(x*ratio) for x in old_size])

        im = cv2.resize(imo, (new_size[1], new_size[0]))

        delta_w = w - new_size[1]
        delta_h = h - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]

        padded_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        dim = target_size
        smallest_dim = min(oh, ow)
        k = 0

        if max_crop:
            while dim+16 <= smallest_dim:
                dim += 16

        screen_coords = []
        img_coords = []
        if darken_bg:
            darken_src = np.ones_like(padded_im)

        time_start = None

        while 1:

            if prevent_smaller_than_target and dim < target_size:
                dim = target_size

            if skip_smaller_than_target and smallest_dim < target_size:
                break

            while dim >= smallest_dim:
                dim -= 16

            fg = padded_im.copy()

            if darken_bg:
                darken = darken_src.copy()

            colour = (255, 0, 0)
            if dim == target_size:
                colour = (0, 255, 0)
            elif dim < target_size:
                colour = (0, 0, 255)

            if lastsave is not None:
                tcolour = (255, 255, 255)
                if lastsave.startswith('FAILED '):
                    tcolour = (0, 0, 255)
                fg = cv2.putText(fg, 'Saved:{}'.format(lastsave), (0, 11), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                fg = cv2.putText(fg, 'Saved:{}'.format(lastsave), (0, 11), font, 0.5, tcolour, 1, cv2.LINE_AA)

            rdim = dim*ratio

            halfdim = rdim//2
            clampx = min(max(lastx, left+halfdim), padded_im.shape[1]-right-halfdim)
            clampy = min(max(lasty, top+halfdim), padded_im.shape[0]-bottom-halfdim)

            orig_x = max(0, int(((clampx-left)/ratio)))
            orig_y = max(0, int(((clampy-top)/ratio)))

            orig_halfdim = dim//2
            orig_clamp_x = min(max(orig_x, orig_halfdim),  imo.shape[1]-orig_halfdim)
            orig_clamp_y = min(max(orig_y, orig_halfdim),  imo.shape[0]-orig_halfdim)

            orig_clamp_dsx, orig_clamp_dsy = orig_clamp_x, orig_clamp_y

            if not grabNow:
                screen_coords = [(clampx, clampy)]
                img_coords = [(orig_clamp_x, orig_clamp_y)]

            if draggStart:
                dsx, dsy = draggStart

                clamp_dsx = min(max(dsx, left+halfdim), padded_im.shape[1]-right-halfdim)
                clamp_dsy = min(max(dsy, top+halfdim),  padded_im.shape[0]-bottom-halfdim)

                orig_dsx = max(0, int(((clamp_dsx-left)/ratio)))
                orig_dsy = max(0, int(((clamp_dsy-top)/ratio)))

                orig_halfdim = dim//2
                orig_clamp_dsx = min(max(orig_dsx, orig_halfdim),  imo.shape[1]-orig_halfdim)
                orig_clamp_dsy = min(max(orig_dsy, orig_halfdim),  imo.shape[0]-orig_halfdim)

                screenDist = math.sqrt(((clampx-clamp_dsx)**2)+((clampy-clamp_dsy)**2))
                if screenDist > rdim//8:
                    slices = (screenDist//(rdim/overlap))+2
                    xs = [int(x) for x in np.linspace(clamp_dsx, clampx, num=int(slices)).tolist()]
                    ys = [int(x) for x in np.linspace(clamp_dsy, clampy, num=int(slices)).tolist()]
                    screen_coords = list(zip(xs, ys))

                    sxs = [int(x) for x in np.linspace(orig_clamp_dsx, orig_clamp_x, num=int(slices)).tolist()]
                    sys = [int(x) for x in np.linspace(orig_clamp_dsy, orig_clamp_y, num=int(slices)).tolist()]
                    img_coords = list(zip(sxs, sys))

            qlen = 0
            try:
                if threadGen is not None:
                    qlen = threadGen.getQueueSize()
            except Exception as e:
                print(e)

            poslist = ','.join(['{}x{}'.format(x-orig_halfdim, y-orig_halfdim) for x, y in img_coords])

            hist = ''
            if historyInd<0:
                hist = 'History Backcrack:{}'.format(abs(historyInd)-1)

            textTemplate = '{} Sample:{} Size:{} Pos:{} ({} crops, overlap:{}) - {} #{} framesPrepared:{} {}'

            templateArgs = (frame_type, n, dim, poslist, len(list(img_coords)), overlap, os.path.basename(f), fi, qlen, hist)

            fg = cv2.putText(fg, textTemplate.format(*templateArgs),
                             (0, fg.shape[0]-11), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            fg = cv2.putText(fg, textTemplate.format(*templateArgs),
                             (0, fg.shape[0]-11), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if not whole_frame_mode:
                for ci, (cx, cy) in enumerate(screen_coords):
                    fg = cv2.rectangle(fg, (int(cx-(rdim//2)), int(cy-(rdim//2))), (int(cx+(rdim//2)), int(cy+(rdim//2))), colour, 1)
                    fg = cv2.putText(fg, '#{}'.format(ci+1), (int(cx-(rdim//2))+5, int(cy-(rdim//2))+15), font, 0.4, colour, 1, cv2.LINE_AA)

                    if darken_bg:
                        darken = cv2.rectangle(darken, (int(cx-(rdim//2)), int(cy-(rdim//2))), (int(cx+(rdim//2)), int(cy+(rdim//2))), (0, 0, 0), -1)

            if darken_bg:
                fg[darken > 0] = fg[darken > 0]//2

            xoffset = 11
            for imgg in lastGrabs:
                if xoffset+11+imgg.shape[0] < fg.shape[0]-35:
                    fg[xoffset+11:xoffset+11+imgg.shape[0], 11:11+imgg.shape[1], :] = imgg
                    xoffset += imgg.shape[0]+10

            if grabNow:
                fn = os.path.basename(f)

                for ix, iy in img_coords[::-1]:
                    n += 1

                    outfile = os.path.join(outdir, '{}_{}_out.png'.format(fn, n))
                    outfiletxt = os.path.join(outdir, '{}_{}_out.txt'.format(fn, n))
                    try:

                        os.path.exists(outdir) or os.makedirs(outdir, exist_ok=True)
                        if whole_frame_mode:
                            imslice = imo
                        else:
                            imslice = imo[iy-orig_halfdim:iy+orig_halfdim, ix-orig_halfdim:ix+orig_halfdim, :]
                            if scale_to_target:
                                imslice = cv2.resize(imslice, (target_size, target_size), interpolation=cv2.INTER_AREA)

                        cv2.imwrite(outfile, imslice)

                        if not skip_txt:
                            open(outfiletxt, 'w').write(' '.join(norm_path))

                        lastGrabs.insert(0, cv2.resize(imslice, (int(100 * (imslice.shape[1]/imslice.shape[0])), 100)))

                        if len(lastGrabs) > 25:
                            lastGrabs.pop()

                        lastsave = outfile
                        print('saved', outfile)

                        grabNow = False
                        if one_click:
                            skip = True
                        if whole_frame_mode:
                            break
                    except Exception as e:
                        print(e)
                        lastsave = 'FAILED '+outfile
                        grabNow = False

            cv2.imshow('imageWindow', fg)
            if time_start is None:
                time_start = time.time()

            k = cv2.pollKey()

            if time_start is not None and slideshow_timeout is not None:
                if abs(time_start-time.time()) > slideshow_timeout:
                    skipNow = True
            if k in (ord('a'),):
                try:
                    if historyInd == 0:
                        history.append((fi, base_file, iframe, frame_type, f, imo, padded_im))
                        historyInd-=1
                    historyInd-=1
                    fi, base_file, iframe, frame_type, f, imo, padded_im = history[historyInd]
                except Exception as e:
                    historyInd+=1
                    pass
            elif k in (ord('y'),):
                grabNow = True
            elif k in (ord('r'),):
                imo = cv2.rotate(imo, cv2.ROTATE_90_CLOCKWISE)
                old_size = imo.shape[:2]
                oh, ow = old_size
                ratiow = float(w)/float(ow)
                ratioh = float(h)/float(oh)
                ratio = min(min(ratiow, ratioh), 1)
                new_size = tuple([int(x*ratio) for x in old_size])
                im = cv2.resize(imo, (new_size[1], new_size[0]))
                delta_w = w - new_size[1]
                delta_h = h - new_size[0]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)
                padded_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif k in (ord('q'),):
                if historyInd == 0:
                    history.append((fi, base_file, iframe, frame_type, f, imo, padded_im))
                skipNow = False
                skip = False
                historyInd = 0
                lastk = k
                break
            elif k != -1 or skip or skipNow:
                if historyInd == 0:
                    history.append((fi, base_file, iframe, frame_type, f, imo, padded_im))
                    
                    while history_max_bytes > 0 and get_history_size() > history_max_bytes:
                        history.popleft()
                    if historyInd < -len(history):
                        historyInd = -len(history)

                skipNow = False
                skip = False

                if historyInd < -1:
                    try:
                        historyInd+=1
                        fi, base_file, iframe, frame_type, f, imo, padded_im = history[historyInd]
                    except Exception as e:
                        historyInd-=1
                        pass
                else:
                    historyInd = 0
                    lastk = k
                    break

        if k in (ord('q'), ord('b')):
            break

    if k == ord('q'):
        break

if skip_seen:
    open(seen_files_path, 'w').write(json.dumps(list(seen)))
