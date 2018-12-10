import sys
sys.path.insert(0,'../')
import platform
from os.path import abspath
from digitclutter import generate, io
from scipy.io import savemat
from progressbar import ProgressBar
import pandas as pd
import numpy as np


'''
Generates an stereoscopic image set inspired by the paper by Spoerer, Kriegeskorte
(https://doi.org/10.1101/133330s)
'''

n_samples = 10
n_letters = 3
fontsize = 300
image_size = (512,512)
rescaled_size = (32,32)
max_occ ={1:3,2:3,3:3,4:3,5:4}

# Do calculations for correctly estimating depth and make the dataset comparable to
# everything else I'm working with


def distancetoangle(object_distance):
    ''' distancetoangle takes the object_distance defined by initializing an object
    and returns the angle needed to adjust vergence of the robot'''
    new_x = object_distance - X_EYES_POSITION
    return - np.arctan(Y_EYES_DISTANCE /
                       np.sqrt(new_x**2 + Y_EYES_DISTANCE**2)) * 360 / (2 * np.pi)



X_EYES_POSITION = 0.062335
Y_EYES_DISTANCE = 0.034000 + 0.034000


# constants
FOC_DIST = 0.5
N_MAX_OCCLUDERS = max_occ[n_letters]
OCC_DIST_TO_FOC = np.zeros([N_MAX_OCCLUDERS])
OCC_DIST = np.zeros([N_MAX_OCCLUDERS])
OCC_SHIFT = np.zeros([N_MAX_OCCLUDERS+1])
SCALING_ARRAY = np.ones([N_MAX_OCCLUDERS+1,2])
#SCALING_ARRAY =  [(1.,1.), (1.25,1.25), (1.5,1.5), (1.875,1.875)]

for i in range(N_MAX_OCCLUDERS):
    OCC_DIST[i] = np.arange(0.4, 0.2, (0.4 - 0.2) / (-1 * N_MAX_OCCLUDERS))[i]
    OCC_DIST_TO_FOC[i] = FOC_DIST - np.arange(0.4, 0.2, (0.4 - 0.2) / (-1 * N_MAX_OCCLUDERS))[i]



for i in range(N_MAX_OCCLUDERS):
    OCC_SHIFT[i+1] = 2*(OCC_DIST_TO_FOC[i] * np.tan(-1.*distancetoangle(FOC_DIST) * (2 * np.pi) / 360.) )
    SCALING_ARRAY[i+1] = (FOC_DIST / OCC_DIST[i], FOC_DIST / OCC_DIST[i])


if platform.system() == 'Windows':
    font_set = ['arial-bold']
else:
    font_set = ['ArialB']

# Generate samples
clutter_list = [generate.sample_clutter(font_set=font_set, n_letters=n_letters, fontsize=fontsize, image_size=image_size, scaling_array = SCALING_ARRAY.tolist()) for i in range(n_samples)]

# Save image set
clutter_list = io.name_files('stereo', clutter_list=clutter_list, prefix='left')
io.save_image_set(clutter_list, 'stereo/{}left.csv'.format(n_letters))

# modify csv data (set background digit to the center)
df = pd.read_csv('stereo/{}left.csv'.format(n_letters), header=None, float_precision='high')

# set focussed object to the middle
df[6] = df[6] * 0.0
df[7] = df[7] * 0.0 
df[7] = 0.025 # five cm below


csv_indices = [7,23,39,55,71]

# set y offset to place objects on a virtual z-pane
for i in range(n_letters):
  df[csv_indices[i]] = -(OCC_SHIFT[i]/2) + 0.025

df.to_csv('stereo/{}left.csv'.format(n_letters), header=None, index=None)

#read image set
clutter_list = io.read_image_set('stereo/{}left.csv'.format(n_letters))
clutter_list = io.name_files('stereo', clutter_list=clutter_list, prefix='left')


# Render images and save as mat file
print('Rendering images...')
bar = ProgressBar(max_value=len(clutter_list))
for i, cl in enumerate(clutter_list):
    cl.render_occlusion()
    bar.update(i+1)
print('Saving mat file...')
fname_list = [cl.fname for cl in clutter_list]
images_dict = io.save_images_as_mat(abspath('stereo/{}left.mat'.format(n_letters)), clutter_list, image_save_size=rescaled_size,
                                    fname_list=fname_list, delete_bmps=False, overwrite_wdir=True)




csv_indices = [6,22,38,54,70]

# modify csv data to shift them for the right eye
df = pd.read_csv('stereo/{}left.csv'.format(n_letters), header=None, float_precision='high')

for i in range(n_letters):
  df[csv_indices[i]] = df[csv_indices[i]] - OCC_SHIFT[i]

df.to_csv('stereo/{}right.csv'.format(n_letters), header=None, index=None)


# Read data again to generate other eye
#read image set
clutter_list = io.read_image_set('stereo/{}right.csv'.format(n_letters))
clutter_list = io.name_files('stereo', clutter_list=clutter_list, prefix='right')

# Render images and save as mat file
print('Rendering images...')
bar = ProgressBar(max_value=len(clutter_list))
for i, cl in enumerate(clutter_list):
    cl.render_occlusion()
    bar.update(i+1)
print('Saving mat file...')
fname_list = [cl.fname for cl in clutter_list]
images_dict = io.save_images_as_mat(abspath('stereo/{}right.mat'.format(n_letters)), clutter_list, image_save_size=rescaled_size,
                                    fname_list=fname_list, delete_bmps=False, overwrite_wdir=True)






# occlusion calculation
# ----

from scipy.io import loadmat, savemat
import imageio, os
import tarfile, sys
 
def untar(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])
    pass

untar('solodigits_square.tar.gz')

eyes = ["left","right"]

for eye in eyes:
  # modify csv data (set background digit to the center)
  df = pd.read_csv('stereo/{}{}.csv'.format(n_letters, eye), header=None, float_precision='high')

  df = df.drop([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,20], axis=1)
  df[2] = df[2] - 1

  df.to_csv('stereo/{}mask{}.csv'.format(n_letters, eye), header=None, index=None)

  #read image set
  clutter_list = io.read_image_set('stereo/{}mask{}.csv'.format(n_letters, eye))
  clutter_list = io.name_files('stereo', clutter_list=clutter_list, prefix='mask{}'.format(eye))


  # Render images and save as mat file
  print('Rendering images...')
  bar = ProgressBar(max_value=len(clutter_list))
  for i, cl in enumerate(clutter_list):
      cl.render_occlusion()
      bar.update(i+1)
  print('Saving mat file...')
  fname_list = [cl.fname for cl in clutter_list]
  images_dict = io.save_images_as_mat(abspath('stereo/{}mask{}.mat'.format(n_letters, eye)), clutter_list, image_save_size=rescaled_size,
                                      fname_list=fname_list, delete_bmps=False, overwrite_wdir=True)



occlusion = {}
for eye in eyes:
  df = pd.read_csv(abspath('stereo/{}{}.csv'.format(n_letters, eye)) , header=None, float_precision='high')
  image_paths = df[0]
  focussed_digits = pd.read_csv(abspath('stereo/{}{}.csv'.format(n_letters, eye)) , header=None, float_precision='high')[5]
  occlusion[eye] = []
  for imgpath, focussed_digit in zip(image_paths, focussed_digits):
    mskpth = imgpath.rsplit('left',-1)[0] + 'mask' + eye + imgpath.rsplit('left',-1)[1]
    imgpth = imgpath.rsplit('left',-1)[0] + eye + imgpath.rsplit('left',-1)[1]
    image = imageio.imread(imgpth + '.bmp', as_gray=True)
    mask = imageio.imread(mskpth + '.bmp', as_gray=True)
    digit = imageio.imread(abspath('solodigits_square/{}'.format(focussed_digit) + '.bmp'), as_gray=True)
    
    a = digit.copy()
    a[a!=119] = 255
    a[a==119] = 0
    original_digit_area = a[a==255].shape[0]
    
    visible_area = (image-mask)[(image-mask)!=0].shape[0]
    occluded_area =original_digit_area - visible_area
    occlusion[eye].append(occluded_area / original_digit_area)
    os.remove(imgpth +'.bmp')
    os.remove(mskpth +'.bmp')
    


occ_avg = (np.array(occlusion["left"]) + np.array(occlusion["right"]))/2.
leftmat = loadmat("stereo/{}left.mat".format(n_letters))
rightmat = loadmat("stereo/{}right.mat".format(n_letters))

leftmat["occlusion"] = np.array(occlusion["left"])
rightmat["occlusion"] = np.array(occlusion["right"])
leftmat["avg_occlusion"] = occ_avg
rightmat["avg_occlusion"] = occ_avg

savemat("stereo/{}left.mat".format(n_letters), leftmat)
savemat("stereo/{}right.mat".format(n_letters), rightmat)


