import cv2
import os
import numpy as np
import random
import sys

print("""
Pass a folder of images as a prameter:

script.py Z:\\somefolder\\somewhere

Space for next image.
Mousewheel for resizing the box
Click to capture the hilighted square

images are saved into \\outdir\\

""")


if len(sys.argv) < 2:
    print('Pass in a source folder as a parameter')
    exit()

files = []
shuffle = False

for source_folder in sys.argv[1:]:
    if source_folder.upper() == 'SHUFFLE':
        shuffle = True
        continue

    if not os.path.isdir(source_folder):
        print('source_folder', source_folder, 'does not exist.')
        continue

    for r, dl, fl in os.walk(source_folder):
        print('source_folder', r, len(fl))
        for f in fl:
            files.append(os.path.join(r, f))

if shuffle:
    random.shuffle(files)

cv2.namedWindow("imageWindow", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("imageWindow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

lastx, lasty = 0, 0
dim = 512
grabNow = False


def on_mouse(event, x, y, flags, param):
    global lastx, lasty, dim, grabNow
    lastx, lasty = (x, y)

    if event == cv2.EVENT_MOUSEWHEEL:
        factor = 16
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            factor = 64

        if flags > 0:
            dim += factor
        elif flags < 0:
            dim -= factor

        dim = max(16, dim)
    elif event == cv2.EVENT_LBUTTONDOWN:
        grabNow = True


cv2.setMouseCallback('imageWindow', on_mouse)

n = 0
font = cv2.FONT_HERSHEY_SIMPLEX
lastsave = None

for f in files:

    imo = cv2.imread(f)
    if imo is None:
        continue

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

    k = 0

    while 1:
        fg = padded_im.copy()
        colour = (255, 255, 255)
        if dim == 512:
            colour = (0, 255, 0)
        elif dim < 512:
            colour = (0, 0, 255)

        if lastsave is not None:
            tcolour = (255, 255, 255)
            if lastsave.startswith('FAILED '):
                tcolour = (0, 0, 255)
            fg = cv2.putText(fg, 'Saved:{}'.format(lastsave), (0, 11), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            fg = cv2.putText(fg, 'Saved:{}'.format(lastsave), (0, 11), font, 0.4, tcolour, 1, cv2.LINE_AA)

        rdim = dim*ratio

        halfdim = rdim//2
        clampx = min(max(lastx, left+halfdim), padded_im.shape[1]-right-halfdim)
        clampy = min(max(lasty, top+halfdim), padded_im.shape[0]-bottom-halfdim)

        orig_x = max(0, int(((clampx-left)/ratio)))
        orig_y = max(0, int(((clampy-top)/ratio)))

        orig_halfdim = dim//2
        orig_clamp_x = min(max(orig_x, orig_halfdim),  imo.shape[1]-orig_halfdim)
        orig_clamp_y = min(max(orig_y, orig_halfdim),  imo.shape[0]-orig_halfdim)

        fg = cv2.putText(fg, 'Sample:{} Size:{} Pos:{}x{}'.format(n, dim, orig_clamp_x-orig_halfdim, orig_clamp_y-orig_halfdim),
                         (0, fg.shape[0]-11), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)

        fg = cv2.putText(fg, 'Sample:{} Size:{} Pos:{}x{}'.format(n, dim, orig_clamp_x-orig_halfdim, orig_clamp_y-orig_halfdim),
                         (0, fg.shape[0]-11), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        fg = cv2.rectangle(fg, (int(clampx-(rdim//2)), int(clampy-(rdim//2))), (int(clampx+(rdim//2)), int(clampy+(rdim//2))), colour, 1)

        if grabNow:
            fn = os.path.basename(f)
            outfile = os.path.join('outdir', '{}_{}_out.png'.format(fn, n))
            try:
                print(outfile)
                os.path.exists('outdir') or os.mkdir('outdir')

                cv2.imwrite(outfile, imo[orig_clamp_y-orig_halfdim:orig_clamp_y+orig_halfdim, orig_clamp_x-orig_halfdim:orig_clamp_x+orig_halfdim, :])

                lastsave = outfile
                print('saved', outfile)
                n += 1
                grabNow = False
            except Exception as e:
                print(e)
                lastsave = 'FAILED '+outfile
                grabNow = False

        cv2.imshow('imageWindow', fg)

        k = cv2.waitKey(1)
        if k != -1:
            break
    if k == ord('q'):
        break
