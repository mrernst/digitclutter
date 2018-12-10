# Stereo Digit Clutter

Contains adapted python scripts for generating digit clutter, digit debris, and stereo digit stimuli as described in the master's thesis 'Recurrent Convolutional Neural Networks for Occluded Object Recognition'

## Prerequisites 

Requires the following python packages, scipy, numpy and Pillow. These should be installed if your using the standard Anaconda distribution. If not these will need to be installed with the following command

```
pip install scipy numpy Pillow pandas
```

Also requires ImageMagick (the code has been tested on version 7). The download and installation instructions can be found [here](http://www.imagemagick.org/script/download.php).

## Basic usage

Firstly, just ensure your files are in a directory on your python path. Having a look at the following working [example](example_script/test_clutter_code.ipynb) should give an idea of how to generate stimulus sets.

A [script](example_script/light_debris_generator.py) is also included that generates 1000 images with the same attributes as the light debris image set described in the paper.

An additional [script](example_script/stereo_generator.py) is included that generates 10 stereoscopic images with the attributes used in the master's thesis 'Recurrent Convolutional Neural Networks for Occluded Object Recognition' set described in the paper.