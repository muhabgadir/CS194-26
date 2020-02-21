# Project 2 : Fun with Filters and Frequencies!

## Usage:

``` sh
python main.py -h
usage: Project 2 for CS 194-26: Fun with filters and frequencies!
Question 1.1 -> python main.py 1.1 -f [Image] -o [OutputImage]
Question 1.2 -> python main.py 1.2 -f [Image] -o [OutputImage]
               (DOF result will be at 'DoG_'OutputImage)
Question 1.3 -> python main.py 1.3 -f [Image] -o [OutputImage]
Question 2.1 -> python main.py 2.1 -f [Image]
             -o [OutputImage] -z [KernelSize] -m [KernelSigma]
Question 2.2 -> python main.py 2.2 -f [LowPassImage]
               -s [HighPassImage] [--gray_one] [--gray_two]
               [-m LowPass sigma] [-g HighPass sigma] -o [OutputImage]
Question 2.3 -> python main.py 2.3 -f [Image]
             -o [OutputImage] -z [KernelSize] -m [KernelSigma]
Question 2.4 -> python main.py 2.4 -f [RegionImage]
             -s [NotRegionImage] -t [MaskImage] [-m KernelSigma] -o [OutputImage]

       [-h] [-f FIRST] [-s SECOND] [-t THIRD] [-o OUTPUT] [-z SIZE] [-m SIGMA1] [-g SIGMA2] [--gray_one] [--gray_two] question

positional arguments:
  question              The number of the question to eval (E.g. 1.1).

optional arguments:
  -h, --help            show this help message and exit
  -f FIRST, --first FIRST
                        The name of the first input image.
  -s SECOND, --second SECOND
                        The name of the second input image.
  -t THIRD, --third THIRD
                        the name of the third input image.
  -o OUTPUT, --output OUTPUT
                        the name of the output image.
  -z SIZE, --size SIZE  Sets the size for the gaussian kernel (Q1, Q2.1, Q2.3-4)
  -m SIGMA1, --sigma_one SIGMA1
                        Sets the sigma for the gaussian kernel (Q1, Q2.1-4)
  -g SIGMA2, --sigma_two SIGMA2
                        Sets the sigma for the gaussian kernel of the second image (Q2.2)
  --gray_one            Toggles greyscale image1(Q2.2)
  --gray_two            Toggles greyscale image2 (Q2.2)
```
