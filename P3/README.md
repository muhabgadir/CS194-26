# Project 3: Face Morphing
# **Make sure that all the images have the same size and ratio or else it won't work!**

## Usage

``` sh
python main.py -h

usage: Project 3 for CS 194-26: Face Morphing
 [-h] [-C IMGC] [-d DEPTH] [-a ALPHA] [--reset] Image ImageB Method Output

positional arguments:
  Image                 The first image or a Population directory path
  Image B               The second image for morphing or the Regex for population images.
  Method                Method to use (Middle, Video, Population, Space).
  Output                Path in which to save the result.

optional arguments:
  -h, --help            show this help message and exit
  -C IMGC, --imageC IMGC
                        The third image for facial space interpolation.
  -d DEPTH, --depth DEPTH
                        Sets the number of interpolation layers for the movie.
  -a ALPHA, --alpha ALPHA
                        Sets the number of interpolation alpha.
  --reset               Asks for point settings again
```
 

There are 4 possible operations:
    * Middle: computes the mid-way face between *Image* and *ImageB* with
      *ALPHA* (optional, default = 0.5 -- mid-way) and stores the result in *out*
    * Video: Computes the morph sequence for *Image* and *ImageB* as it stores
      the final video in *out*. Also, ignore the QTimer warning if it shows. (Please use an *ffmpeg* writer compatible format
      -- *.gif* doesn't work. If you want to create a *.gif*, generate the video
      with a *.mp4* format and then run the following bash snippet:
      ``` sh
      ffmpeg -i video.mp4 -f gif video.gif
      ```
      ) 
    * Population: Calculate the mean face of a population (with the DTU format
      (.asf files with the same structure)) given the directory of
      the database (input to *Image*) and a regex for image selection (input to
      *ImageB*). Result will be stored in the *out* input. (ex:
      ``` sh
      python main.py /home/dewey/Misc/db/ ".*-1m\.jpg" Population pop_m.jpg
      ```
      This selects all the male pictures from the DTU database stored somewhere
      in my home directory.)
    * Space: Computes a video of *Image* being moved around the interpolation
      space between *ImageB* and *IMGC*. The result will be in *out*.

For all the operations besides *Population*, I require *.p* files. These are
files in which I saved using the *Pickle* library in python. All the points for
the images generated are there. The program reads an input image and looks for
its *.p* file. If it doesn't have one, it asks for you to input them as 58
points in the image using *ginput*. If you have your set of points and are able
to have it structured as a *np.array* in python, just do the following:

``` python
import pickle
pickle.dump(your_structure, open("image_name.p", "wb"))
```

And then run my program. If you have an *.asf* file with the same structure as
the ones in DTU, you can take advantage of the *Population* method and give the
name of your file as regex. It will, then, locate the *.asf* file and interpret
it, creating a *.p* file for the output, which will be the same image.
