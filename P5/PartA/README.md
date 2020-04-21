# Project 5A: Image Rectification and Mosaicing

Use the following command to compute a rectification from an image to another:

``` python
wrp = image_warpper(image1_name, image2_name, N, isgray)
wrp.compute_morph(exponent) # exponent for the blend alpha function
wrp.show_result()
wrp.save_result(filename) # filename to save the picture to
```

If the image is one-channeled, make isgray true! the "N" is supposed to be the
number of correspondences to input. If you just want to rectify the image, make
image1 just a black picture and draw the correspondences when prompted. If you
want to redo the warping, delete the .p files in which I save the points so you
don't have to reset the points for the mosaic

If some IndexError happens, just retry until it works. It can compute some weir
transform that doesn't fit the canvas (new image).
