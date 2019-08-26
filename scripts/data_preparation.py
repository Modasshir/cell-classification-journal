import os
import random
import shutil
import xml.etree.ElementTree as ET
from glob import glob
from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np
import pandas as pd
import progressbar
from PIL import Image
from tqdm import tqdm

np.random.seed(1234)


# useful functions

def get_patch_coordinates(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    markerlist = dict()
    for child in root:
        if child.tag == 'Image_Properties':
            for grandchild in child:
                tempstr = grandchild.text
                filename = tempstr[tempstr.rfind(' ') + 1:]
        if child.tag == 'Marker_Data':
            for gchild in child:
                if gchild.tag == 'Marker_Type':
                    for ggchild in gchild:
                        if ggchild.tag == 'Type':
                            markertype = int(ggchild.text)
                            markerlist[markertype] = []
                        if ggchild.tag == 'Marker':
                            xval = int(ggchild[0].text)
                            yval = int(ggchild[1].text)
                            (markerlist[markertype]).append([xval, yval])
    return filename, markerlist


def load_directory_of_image(dirstr):
    outlist = []
    namelist = []
    onlyfiles = [join(dirstr, f) for f in listdir(dirstr)
                 if isfile(join(dirstr, f)) and f.endswith(".gif")]
    for f in onlyfiles:
        #        p = Image.open(f).resize((w, h), Image.ANTIALIAS)
        p = Image.open(f)
        npa = np.array(p.convert('I')).astype('uint8')
        #        print npa.shape
        outlist.append(npa.reshape(1, npa.shape[0], npa.shape[1]))
        namelist.append(f)
    return namelist, outlist


def create_txt_files(image_files, train_split=0.95, val_split=0.05):
    random.shuffle(x=image_files)
    labels = [x.split('/')[-2] for x in image_files]
    unique_labels = set(labels)
    label_encoder = dict()
    idx = 0
    for item in unique_labels:
        label_encoder[item] = idx
        idx += 1
    data = [x + ' ' + str(label_encoder[x.split('/')[-2]])
            for x in image_files]
    with open('data.txt', 'w') as f:
        f.writelines('\n'.join(data))
    with open('train.txt', 'w') as f:
        f.writelines(
            '\n'.join(data[1:int(len(data) * (train_split - val_split))]))

    with open('val.txt', 'w') as f:
        f.writelines('\n'.join(
            data[int(len(data) * (train_split - val_split)):int(len(data) * train_split)]))

    with open('test.txt', 'w') as f:
        f.writelines('\n'.join(data[int(len(data) * train_split):]))
    with open('labels.txt', 'w') as f:
        f.writelines('\n'.join(label_encoder.keys()))


# # # Creating Important classifications, i.e. type one and two


def create_txt_files_with_suffix(image_files, train_split=0.8, val_split=0.2, suffix='imp'):
    random.shuffle(x=image_files)
    custom_le = {'zero': 0, 'one': 1, 'two': 2}
    data = []
    for x in image_files:
        cell_type = x.split('/')[-2]
        if cell_type in ['one', 'two']:
            data.append(x + ' ' + str(custom_le[cell_type]))
        else:
            data.append(x + ' ' + '0')
    with open('data' + suffix + '.txt', 'w') as f:
        f.writelines('\n'.join(data))
    with open('train' + suffix + '.txt', 'w') as f:
        f.writelines(
            '\n'.join(data[1:int(len(data) * (train_split - val_split))]))

    with open('val' + suffix + '.txt', 'w') as f:
        f.writelines('\n'.join(
            data[int(len(data) * (train_split - val_split)):int(len(data) * train_split)]))

    with open('test' + suffix + '.txt', 'w') as f:
        f.writelines('\n'.join(data[int(len(data) * train_split):]))
    with open('labels' + suffix + '.txt', 'w') as f:
        f.writelines('\n'.join(custom_le.keys()))


def get_image_data(txt, save=False, file_name=None):
    with open(txt, 'r') as f:
        content = f.readlines()
    df = pd.DataFrame()
    df['dir'] = [x.split(' ')[0].strip() for x in content]
    df['label'] = [x.split(' ')[1].strip() for x in content]
    df['label'] = df['label'].astype(np.uint8)
    if save:
        if file_name is None:
            print('you must provide file name, not saving')
            return df
        df.to_pickle(path=file_name + '.gz', compression='gzip')
    return df


def create_folderwise_data(txt, folder):
    print('\ncleaning target directory')
    files = glob('../patches/' + folder + '/*/*')
    for f in files:
        os.remove(f)
    print('creating ' + folder + ' folder')
    df = get_image_data(txt)
    df_dir = []
    for i in range(df.dir.shape[0]):
        parts = df.dir[i].split('/')
        parts.insert(-2, folder)
        df_dir.append('/'.join(parts))
    df['new_dir'] = df_dir
    print(df.head())

    for i in tqdm(range(df.dir.shape[0])):
        try:
            shutil.move(src=df.dir[i], dst=df.new_dir[i])
        except:
            print('../patches/' + folder + '/' + df.dir[i].split('/')[-2])
            os.makedirs('../patches/' + folder +
                        '/' + df.dir[i].split('/')[-2])
            shutil.move(src=df.dir[i], dst=df.new_dir[i])


# creating essential folders for data
print('\ncleaning and creating essential folders')
if os.path.isdir('../patches'):
    shutil.rmtree('../patches')

os.makedirs('../patches/one')
os.makedirs('../patches/two')
os.makedirs('../patches/four')
os.makedirs('../patches/five')
os.makedirs('../patches/zero')
os.makedirs('../patches/train')
os.makedirs('../patches/test')
os.makedirs('../patches/val')

marker_list_path = '../annotations/*.xml'
markerfilelist = glob(marker_list_path)
random.shuffle(x=markerfilelist)
print('Number of annotation files: ' + str(len(markerfilelist)))

markerdict = dict()
for f in markerfilelist:
    filename, markerlist = get_patch_coordinates(f)
    markerdict[filename] = markerlist

imagenames, imagedata = load_directory_of_image('../images')

print('creating type wise patches')
pathname = '../patches/'
badcount = 0

for k in tqdm(range(len(imagenames))):
    IN = imagenames[k].split('/')[2]
    ID = imagedata[k][0]
    filestr = IN[IN.rfind('\\') + 1:-4] + '.tif'
    if filestr in markerdict:
        MD = markerdict[filestr]
        for markertype in MD:
            typedict = MD[markertype]
            for pos in typedict:
                xpos = pos[1]
                ypos = pos[0]
                if xpos - 16 < 0:
                    xmin = 0
                    xmax = 33
                elif xpos + 16 > ID.shape[0]:
                    xmax = ID.shape[0] + 1
                    xmin = ID.shape[0] - 32
                else:
                    xmin = xpos - 16
                    xmax = xpos + 17
                if ypos - 16 < 0:
                    ymin = 0
                    ymax = 33
                elif ypos + 16 > ID.shape[1]:
                    ymax = ID.shape[1] + 1
                    ymin = ID.shape[1] - 32
                else:
                    ymin = ypos - 16
                    ymax = ypos + 17
                tempcont = ID[xmin:xmax, ymin:ymax]
                im = Image.fromarray(tempcont)
                folder_name = {'1': 'one',
                               '2': 'two',
                               '3': 'three',
                               '4': 'four',
                               '5': 'five'}
                im.save(join(pathname, folder_name[str(markertype)],
                             filestr[:-4] + '-' + str(markertype) + '_' + str(xpos) + '_' + str(ypos) + '.jpg'))
                # Image.fromarray(tempcont).save(
                #     join(pathname, str(markertype), filestr[:-4] + '-' + str(markertype) + '_' + str(
                #         xpos) + '_' + str(ypos) + '.gif'))
    else:
        badcount += 1

# print(badcount)
print('finished creating type wise patches')

# IN0 = imagenames[0].split('/')[2]
# IN0[IN0.rfind('\\')+1:-4]+'.tif'


print('creating negative class patches')
pathname = '../patches/zero/'
targetcount = 10
for k in tqdm(range(len(imagenames))):
    IN = imagenames[k].split('/')[2]
    ID = imagedata[k][0]
    w, h = ID.shape
    foundcount = 0
    filestr = IN[IN.rfind('\\') + 1:-4] + '.tif'
    if filestr in markerdict:
        while foundcount < targetcount:
            MD = markerdict[filestr]
            xpos = randint(0, w - 1)
            ypos = randint(0, h - 1)
            # print 'xpos,ypos=',xpos,',',ypos
            foundmatch = False
            for markertype in MD:
                typedict = MD[markertype]
                for pos in typedict:
                    tempx = pos[1]
                    tempy = pos[0]
                    if abs(xpos - tempx) < 32 and abs(ypos - tempy) < 32:
                        # print markertype,tempx,tempy
                        foundmatch = True
                        break
                if foundmatch:
                    break
            if not foundmatch:
                foundcount += 1
                if xpos - 16 < 0:
                    xmin = 0
                    xmax = 33
                elif xpos + 16 > ID.shape[0]:
                    xmax = ID.shape[0] + 1
                    xmin = ID.shape[0] - 32
                else:
                    xmin = xpos - 16
                    xmax = xpos + 17
                if ypos - 16 < 0:
                    ymin = 0
                    ymax = 33
                elif ypos + 16 > ID.shape[1]:
                    ymax = ID.shape[1] + 1
                    ymin = ID.shape[1] - 32
                else:
                    ymin = ypos - 16
                    ymax = ypos + 17
                tempcont = ID[xmin:xmax, ymin:ymax]
                name = pathname + filestr[:-4] + '-0_' + \
                    str(xpos) + '_' + str(ypos) + '.jpg'
                Image.fromarray(tempcont).save(name)
# print(foundcount)
print('finished creating negative class patches')

image_files = glob('../patches/*/*')
create_txt_files(image_files)

create_txt_files_with_suffix(image_files)

create_folderwise_data('./train.txt', 'train')
create_folderwise_data('./test.txt', 'test')
create_folderwise_data('./val.txt', 'val')

print('\ncleaning unwanted files and folders')
shutil.rmtree('../patches/one')
shutil.rmtree('../patches/two')
shutil.rmtree('../patches/four')
shutil.rmtree('../patches/five')
shutil.rmtree('../patches/zero')

[os.remove(x) for x in glob('./*.txt')]

print('data preparation finished.')
