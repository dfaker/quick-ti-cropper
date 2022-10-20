# quick-ti-cropper
Quick manual cropping utility to grab 1:1 ratio sections from a folder of images

# cropper.py

## requirements

`pip install opencv-python`

## usage

Pass a folder of images as a parameter:
`cropper.py Z:\\somefolder\\somewhere`

or to shuffle:
`cropper.py shuffle Z:\\somefolder\\somewhere`

or to shuffle multiple folders:
`cropper.py shuffle Z:\\somefolder\\somewhere Z:\\someotherfolder\\somewhere`

Pass a folder of images as a prameter:

cropper.py [OptionalArgs] Z:\\somefolder\\somewhere

 - **Right Click** or **Space** for next image.
 - **Mousewheel** for resizing the box, Shift tot resize faster
 - **Left Click** to capture the hilighted square
 - **Left Click and Drag** to evenly spread a line of crops.
 - **Ctrl and Mousewheel** Change the overlap of crops when dragging.

Optional Args:

 - **SHUFFLE** - Shuffle Images.
 - **ONECLICK** - Jump to next image on every click.
 - **LARGESTFIRST** - Sort files by size, largest to smallest.
 - **MAXCROP** - Initialize on each image with the largest possible crop size.
 - **DARKENBG** - Darken the unselected areas of the image in preview.

images are saved into .\\outdir\\

![image](https://user-images.githubusercontent.com/35278260/196969198-acc055e0-a77d-4db4-9e97-f1e836bd2f90.png)

# getframes.py

## requirements

ffmpeg

## usage

Pass one or more video files a parameter:

`getframes.py Z:\\somefolder\\video1.mp4`

will extract a frame once every 60 seconds. and place it in the '.\frames\' directory
