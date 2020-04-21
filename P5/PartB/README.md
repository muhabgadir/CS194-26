# Project 5B: Auto stitching of mosaics

To run this project, you need to use the module inside the main.py file. You
should use it as follows:

``` python
mop = mops(image1_name, image2_name, edge_discard, isgray) # if images are one-channeled, make isgray = True
mop.compute_features(False) #True if you want the corresp. features shown
mop.blend_images(filename) #filename to save the new blended image
```

Please, don't use the following format for the image name: "./name.format".
Instead, use "name.format". If you run the algorithm, it will save the
calculated points in a .p2 file. If you want it to run again with a different
edge discard, please delete these files.

Some unpredictable errors might occur: some IndexErrors or feature map
assertions. If these happen, please delete the .p2 files and try with another
edge_discard (mostly lowering it) since it's probably calculating bad edges. The
edge calculation part can get slow if your image is big. It does with the
example images once we start stitching them and they get large. If it gets
really slow (around 20000 edges it starts getting unbearably slow), try
increasing the edge_discard. I used the following edge_discards for my results:

Rio (5 images) -> 150
Shanghai (4 images) -> 150
Goldengate (4 images) -> 100

To compute a panorama, stack the command above. Please, make sure *image2* is on
the right of *image1*, because it will be warpped assuming that.
