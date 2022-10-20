import cv2
import os
import numpy as np
import random
import sys
import os
import math

print("""
Pass a folder of images as a prameter:

cropper.py [OptionalArgs] Z:\\somefolder\\somewhere

Right Click or Space for next image.
Mousewheel for resizing the box, Shift tot resize faster
Left Click to capture the hilighted square
Left Click and Drag to evenly spread a line of crops.
Ctrl and Mousewheel Change the overlap of crops when dragging.

Optional Args:

SHUFFLE - Shuffle Images.
ONECLICK - Jump to next image on every click.
LARGESTFIRST - Sort files by size, largest to smallest.
MAXCROP - Initialize on each image with the largest possible crop size.
DARKENBG - Darken the unselected areas of the image in preview.

images are saved into .\\outdir\\

""")


if len(sys.argv) < 2:
    print('Pass in a source folder as a parameter')
    exit()

files = []
shuffle = False
one_click = False
largest_first = False
max_crop = False
darken_bg = False

for source_folder in sys.argv[1:]:
    if source_folder.upper() == 'SHUFFLE':
        shuffle = True
        continue
    if source_folder.upper() == 'ONECLICK':
        one_click = True
        continue
    if source_folder.upper() == 'LARGESTFIRST':
        largest_first = True
        continue
    if source_folder.upper() == 'MAXCROP':
        max_crop = True
        continue
    if source_folder.upper() == 'DARKENBG':
        darken_bg = True
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
if largest_first:
    files = sorted(files, key=lambda x: os.stat(x).st_size, reverse=True)

cv2.namedWindow("imageWindow", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("imageWindow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

lastx, lasty = 0, 0
coord_list = []
dim = 512
grabNow = False
skipNow = False
draggStart = None
overlap = 1.25

def on_mouse(event, x, y, flags, param):
    global lastx, lasty, dim, grabNow, skipNow, draggStart,coord_list,overlap
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


cv2.setMouseCallback('imageWindow', on_mouse)

n = 0
font = cv2.FONT_HERSHEY_SIMPLEX
lastsave = None

for f in files:
    
    norm_path = os.path.normpath(f)
    norm_path = norm_path.split(os.sep)[2:-1]

    print(norm_path)

    skip=False
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

    dim=512
    smallest_dim = min(oh, ow)
    k = 0

    if max_crop:
        while dim+16 <= smallest_dim:
            dim += 16

    screen_coords = []
    img_coords = []
    if darken_bg:
        darken_src = np.ones_like(padded_im)

    while 1:

        while dim >= smallest_dim:
            dim -= 16

        fg = padded_im.copy()
        
        if darken_bg:
            darken = darken_src.copy()
        
        colour = (255, 0, 0)
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
            if screenDist > rdim//4:
                slices = (screenDist//(rdim/overlap))+2
                xs = [int(x) for x in np.linspace(clamp_dsx, clampx, num=int(slices)).tolist()]
                ys = [int(x) for x in np.linspace(clamp_dsy, clampy, num=int(slices)).tolist()]
                screen_coords = list(zip(xs, ys))

                sxs = [int(x) for x in np.linspace(orig_clamp_dsx, orig_clamp_x, num=int(slices)).tolist()]
                sys = [int(x) for x in np.linspace(orig_clamp_dsy, orig_clamp_y, num=int(slices)).tolist()]
                img_coords = list(zip(sxs, sys))
                print(img_coords)

        poslist = ','.join(['{}x{}'.format(x-orig_halfdim, y-orig_halfdim) for x, y in img_coords])

        fg = cv2.putText(fg, 'Sample:{} Size:{} Pos:{} ({} crops, overlap:{}) - {}'.format(n, dim, poslist, len(list(img_coords)), overlap, f),
                         (0, fg.shape[0]-11), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)

        fg = cv2.putText(fg, 'Sample:{} Size:{} Pos:{} ({} crops, overlap:{}) - {}'.format(n, dim, poslist, len(list(img_coords)), overlap, f),
                         (0, fg.shape[0]-11), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for cx, cy in screen_coords:
            fg = cv2.rectangle(fg, (int(cx-(rdim//2)), int(cy-(rdim//2))), (int(cx+(rdim//2)), int(cy+(rdim//2))), colour, 1)
            if darken_bg:
                darken = cv2.rectangle(darken, (int(cx-(rdim//2)), int(cy-(rdim//2))), (int(cx+(rdim//2)), int(cy+(rdim//2))), (0, 0, 0), -1)

        if darken_bg:
            fg[darken>0] = fg[darken>0]//2

        if grabNow:
            fn = os.path.basename(f)
            print(img_coords)
            for ix, iy in img_coords:
                print(ix,iy)
                n += 1

                outfile = os.path.join('outdir', '{}_{}_out.png'.format(fn, n))
                outfiletxt = os.path.join('outdir', '{}_{}_out.txt'.format(fn, n))
                try:
                    print(outfile)
                    os.path.exists('outdir') or os.mkdir('outdir')

                    cv2.imwrite(outfile, imo[iy-orig_halfdim:iy+orig_halfdim, ix-orig_halfdim:ix+orig_halfdim, :])
                    open(outfiletxt, 'w').write(' '.join(norm_path))

                    lastsave = outfile
                    print('saved', outfile)
                    
                    grabNow = False
                    if one_click:
                        skip = True
                except Exception as e:
                    print(e)
                    lastsave = 'FAILED '+outfile
                    grabNow = False

        cv2.imshow('imageWindow', fg)

        k = cv2.waitKey(1)
        if k != -1 or skip or skipNow:
            skipNow = False
            break

    if k == ord('q'):
        break
