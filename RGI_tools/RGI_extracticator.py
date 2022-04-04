import tarfile
import shutil
import os
from os.path import join, getsize
from tqdm import tqdm
# this loop runs through all subdirectories RGI60-n
# opens each tarball to desired folder, currently ~/data/glac/RGI_TOPO/dump_1
# '/home/sa42/data/glac/RGI_TOPO/'
# '/path/to/tar'
rootdir = '/data/fast1/glacierml/RGI/'
ext = ('.tar')
print('Opening tar balls')
for subdir, dirs, files in tqdm(os.walk(rootdir, topdown = False, onerror = None, followlinks = True)):
        for file in files:
            if file.endswith(ext):
                filepath = os.path.join(subdir,file)
                tar = tarfile.open(filepath)
                tar.extractall(path='/data/fast1/glacierml/RGI/dump1/')
                tar.close()
                
                
# this loop runs through all subdirectories RGI60-n
# opens each tarball to desired folder, currently ~/data/glac/RGI_TOPO/dump_1
# '/home/sa42/data/glac/RGI_TOPO/'
# '/path/to/tar'
# '/path/to/dump_1/'

print('Extracting glacier information')
rootdir = '/data/fast1/glacierml/RGI/dump1/'
ext = ('.tar.gz')
for subdir, dirs, files in tqdm(os.walk(rootdir, topdown = False, onerror = None, followlinks = True)):
        for file in files:
            if file.endswith(ext):
                filepath = os.path.join(subdir,file)
                tar = tarfile.open(filepath)
                tar.extractall(path='/data/fast1/glacierml/RGI/dump2/')
                tar.close()
                
                
                

# this loop will need to extract dem.tif files
# decided to target MAPZEN, according to the documentation this has the most coverage
# able to run through all mapzen files and extract them, but they are overwritten
# need to rename the file after the folder containing it....
# '/home/sa42/data/glac/RGI_TOPO/dump_2/'
# '/path/to/dump_2/'

# '/home/sa42/data/glac/RGI_TOPO/dump_2/'

      

print('Extracting DEM data')
rootdir = '/data/fast1/glacierml/RGI/dump2/'

# 'MAPZEN'
rect = ('MAPZEN')
ext = ('dem.tif')

# '/home/sa42/data/glac/RGI_TOPO/extracted_dem_files/MAPZEN/'
mapzen = '/data/fast1/glacierml/RGI/extracted_dem_files/MAPZEN/'
for subdir, root, files in tqdm(os.walk(rootdir, topdown = True, onerror = None, followlinks = False)):
    if subdir.endswith(rect):
        for file in files:
            if file.endswith(ext):
                filepath = os.path.join(subdir,file)
                filepath_left = filepath[37:]
                RGI_name = filepath_left [:14]
                shutil.copy(filepath, mapzen + RGI_name + '_' + 'dem.tif')

                
                
                
# OUTLINE LOOP

# this loop is rooted to where tarballs are untarred, currently '/home/sa42/data/glac/RGI_TOPO/dump_2'
# this loop runs through extracted tarballs and opens the .tar.gz files inside
# sent to 'destination'
# '/path/to/dump_2'
# '/home/sa42/data/glac/RGI_TOPO/dump_2'

print('Extracting glacier mask data')

desired_file = ('glacier_mask')
desired_extension = ('.tif')

# '/home/sa42/data/glac/RGI_TOPO/extracted_outlines/'
destination = ('/data/fast1/glacierml/RGI/extracted_mask_files/')
# origin = 
# 'outlines.shx'
extracted_file = ('glacier_mask.tif')
for subdir, dirs, files in os.walk(rootdir, topdown = False, onerror = None, followlinks = True):
    for file in files:
        if file.endswith(desired_extension):
            if file.startswith(desired_file):
                filepath = os.path.join(subdir,file)
                filepath_left = filepath[37:]
                filepath_reg = filepath_left [:14]
#                 print(join(subdir,extracted_file))
#                 print(destination + filepath_reg + '_' + extracted_file)

#                 tar = tarfile.open(filepath)
#                 tar.extract(extracted_file, destination)
#                 tar.close()
               
                shutil.move(
                    join(subdir,extracted_file), destination + filepath_reg + '_' + extracted_file)
#                 shutil.move(
#                     destination + extracted_file, destination + filepath_reg + '_' + extracted_file)
                

