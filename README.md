# quick-ti-cropper
Quick manual cropping utility to grab 1:1 ratio sections from a folder of images

# cropper.py

## requirements

`pip install opencv-python`

Reading Video files requires ffmpeg.

## usage

Pass a folder of images or videos, or single files  as a prameter:

`cropper.py [OptionalArgs] Z:\\somefolder\\somewhere`

`cropper.py [OptionalArgs] Z:\\somefolder\\somewhere\\test.png`

`cropper.py [OptionalArgs] Z:\\somefolder\\somewhere\\test.mp4`

or to shuffle:
`cropper.py shuffle Z:\\somefolder\\somewhere`

or to shuffle multiple folders:
`cropper.py shuffle Z:\\somefolder\\somewhere Z:\\someotherfolder\\somewhere`

Pass a folder of images as a prameter:

cropper.py [OptionalArgs] Z:\\somefolder\\somewhere


## Keys:

- **Right Click** or **Space** for next image.
- **Mousewheel** for resizing the box, Shift tot resize faster
- **Left Click** to capture the hilighted square
- **Left Click and Drag** to evenly spread a line of crops.
- **Ctrl and Mousewheel** Change the overlap of crops when dragging.
- **B** Skip to the next video during video frame cropping.
- **A** and **D** to go back and forwards in frame history
- **Right Click** or Space for next image.
- **Mousewheel** for resizing the box, Shift tot resize faster
- **Left Click** to capture the hilighted square
- **Left Click and Drag** to evenly spread a line of crops.
- **Ctrl and Mousewheel** Change the overlap of crops when dragging.
- **B** to skip to the next video during video frame cropping.
- **R** to rotate the image clockwise.

## Optional Args:

### General options:
- **SHUFFLE** - Shuffle Images.
- **ONECLICK** - Jump to next image on every click.
- **LARGESTFIRST** - Sort files by size, largest to smallest.
- **MAXCROP** - Initialize on each image with the largest possible crop size.
- **DARKENBG** - Darken the unselected areas of the image in preview.
- **WHOLEFRAME** - don't crop, take the whole frame
- **SLIDESHOW_TIMEOUT** - timeout to skip to next image automatically
- **SKIPTXT** - Do not create caption .txt files
- **NOSCALE** - Do not scale final images down to TARGET_SIZExTARGET_SIZE
- **NOIMAGES** - Skip display of image files.
- **NOVIDEO** - Skip display of video frames.
- **MAX_FILE_SIZE=N** - Do not load files over N MB in size.
- **TARGET_SIZE=N** - Set the target dimension in pixels (Default 512)
- **OUTDIR=DIR** - Output to a directory called DIR
    **ALLOW_SMALLER_CROP** - Allow crops smaller than TARGET_SIZExTARGET_SIZE

### Video Options:
- **IFRAME_GAP=N** - When cropping video frames, wait at least N seconds between Key frames (Default 10).
- **VIDEO_START=N** - When cropping video frames, Sart this many seconds or HH:MM:SS into the video

### Cache Limits:
- **FRAME_FWD_CACHE=N** - The maximum number of bytes to use for a forward frame cache when cropping video frames (Default 1GB).
- **FRAME_HIST_CACHE=N** - The maximum number of bytes to use to keep images in history so they can be skipped back to (Default 1GB).

 
images are saved into .\\outdir\\

![image](https://user-images.githubusercontent.com/35278260/196969198-acc055e0-a77d-4db4-9e97-f1e836bd2f90.png)

![image](https://user-images.githubusercontent.com/35278260/197090519-5381d622-47b8-45ac-bdc6-be30f6f7f18a.png)


# getframes.py

## requirements

ffmpeg

## usage

Pass one or more video files a parameter:

`getframes.py Z:\\somefolder\\video1.mp4`

will extract a frame once every 60 seconds. and place it in the '.\frames\' directory
