# Images of the Russian Empire: Colorizing the Prokudin-Gorskii photo collection
### Author: Guilherme Gomes Haetinger
### University of California, Berkeley
### CS 194 - 26
### Jan 2020

## Execution
The program only has one file: *main.py*, which can be run in the following format:

```
usage: main.py [-h] [--NCC] [--naive] [--autocrop] [--full-autocrop]
               InputFile OutputFile

Image recoloring through channel alignment.

positional arguments:
  InputFile        The name of the input image.
  OutputFile       The name of the output image.

optional arguments:
  -h, --help       show this help message and exit
  --NCC            Sets image aligning method to Normalized Cross Correlation.
                   (Default: Sum of Squared Differences)
  --naive          Sets image to be aligned exhaustively instead of optimized
                   with the pyramid method.
  --autocrop       Toggles auto-cropping method that enables better
                   performance and black border removal.
  --full-autocrop  Toggles auto-cropping method for the initial image state
                   and the final image state.
```

*I'm a concurrent enrollment student and I don't have an instructional account to upload the HTML page. The page is within this submission as well!**
