# quick-ti-cropper
Quick manual cropping utility to grab 1:1 ratio sections from a folder of images

# requirements

`pip install opencv-python`

# usage

Pass a folder of images as a prameter:
`cropper.py Z:\\somefolder\\somewhere`

or to shuffle:
`cropper.py shuffle Z:\\somefolder\\somewhere`

or to shuffle multiple folders:
`cropper.py shuffle Z:\\somefolder\\somewhere Z:\\someotherfolder\\somewhere`

**Space** for next image.
**Mousewheel** for resizing the box
**Click** to capture the hilighted square

images are saved into `\outdir\`

![image](https://user-images.githubusercontent.com/35278260/195849586-7ada7249-275f-4dd6-9069-5518bfab46ac.png)
