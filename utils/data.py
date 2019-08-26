import os
from glob import glob
import numpy as np
import pandas as pd
import progressbar
from PIL import Image, ImageDraw
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from misc_utils import *


def get_data(img_width=32, img_height=32, channel=1, getDfs=False):
    train_df, val_df, test_df = get_only_data_lists()

    print('loading training data' + '.' * 10)
    x_train, y_train3, y_train5 = get_partial_data(
        train_df, img_width, img_height, channel)
    print('loading validation data' + '.' * 10)
    x_val, y_val3, y_val5 = get_partial_data(
        val_df, img_width, img_height, channel)
    print('loading testing data' + '.' * 10)
    x_test, y_test3, y_test5 = get_partial_data(
        test_df, img_width, img_height, channel)
    if getDfs:
        return x_train, y_train3, y_train5, x_val, y_val3, y_val5, x_test, y_test3, y_test5, train_df, val_df, test_df
    else:
        return x_train, y_train3, y_train5, x_val, y_val3, y_val5, x_test, y_test3, y_test5


def get_partial_data(df, img_width=32, img_height=32, channel=1):
    patches = np.ndarray(shape=(len(df.label), img_width,
                                img_height, channel), dtype=np.float32)
    gray = True if channel == 1 else False
    bar = progressbar.ProgressBar(maxval=df.shape[0])
    bar.start()
    for i in range(df.shape[0]):
        patches[i, :, :, :] = img_to_array(load_img(df.dir.iloc[i], grayscale=gray,
                                                    target_size=(img_width, img_height)))
        bar.update(i)
    bar.finish()
    label5 = to_categorical(np.vstack(df['5_classes'].values))
    label3 = to_categorical(np.vstack(df['3_classes'].values))
    return patches, label3, label5


# def get_only_data_lists():
#     df = pd.DataFrame(columns=['dir', 'label'])
#     df.dir = glob('../patches/*/*/*')
#     df.label = [x.split('/')[-2] for x in df.dir]
#     class_dict = {'one': 1,
#                   'two': 2,
#                   'four': 3,
#                   'five': 4,
#                   'zero': 0}
#     df['5_classes'] = df.label.map(class_dict)

#     class_dict = {'one': 1,
#                   'two': 2,
#                   'four': 0,
#                   'five': 0,
#                   'zero': 0}
#     df['3_classes'] = df.label.map(class_dict)

#     df = shuffle(df)
#     np.random.seed(1234)
#     train_df, test_df = train_test_split(df, train_size=0.8)
#     train_df, val_df = train_test_split(train_df, train_size=0.8)
#     return train_df, val_df, test_df

def get_only_data_lists():

    train_df = pd.DataFrame(columns=['dir', 'label'])
    train_df.dir = glob('../patches/train/*/*')
    train_df.label = [x.split('/')[-2] for x in train_df.dir]
    val_df = pd.DataFrame(columns=['dir', 'label'])
    val_df.dir = glob('../patches/val/*/*')
    val_df.label = [x.split('/')[-2] for x in val_df.dir]
    test_df = pd.DataFrame(columns=['dir', 'label'])
    test_df.dir = glob('../patches/test/*/*')
    test_df.label = [x.split('/')[-2] for x in test_df.dir]

    class_dict = {'one': 1,
                  'two': 2,
                  'four': 3,
                  'five': 4,
                  'zero': 0}
    train_df['5_classes'] = train_df.label.map(class_dict)
    val_df['5_classes'] = val_df.label.map(class_dict)
    test_df['5_classes'] = test_df.label.map(class_dict)

    class_dict = {'one': 1,
                  'two': 2,
                  'four': 0,
                  'five': 0,
                  'zero': 0}

    train_df['3_classes'] = train_df.label.map(class_dict)
    val_df['3_classes'] = val_df.label.map(class_dict)
    test_df['3_classes'] = test_df.label.map(class_dict)

    return train_df, val_df, test_df


def get_segmentation_data(dir):
    img_width = 512
    img_height = 672
    img_dir = glob(dir)

    images = np.ndarray(shape=(len(img_dir), img_width,
                               img_height, 1), dtype=np.float32)
    masks = np.ndarray(shape=(len(img_dir), img_width,
                              img_height, 1), dtype=np.float32)
    for i in range(len(img_dir)):
        mask_image = '../segmentation/mask/' + img_dir[i].split('/')[-1]

        if os.path.isfile(mask_image):
            images[i, :, :, :] = img_to_array(load_img(img_dir[i], grayscale=True,
                                                       target_size=(img_width, img_height)))
            masks[i, :, :, :] = img_to_array(load_img(mask_image, grayscale=True,
                                                      target_size=(img_width, img_height)))
    return images, masks


def get_masked_image(file_name, mdict, rad_dict, scale=1.):
    if type(file_name) == str:
        img = Image.open('../png_images/' + file_name, mode='r')
    else:
        img = Image.fromarray(file_name)
    img = img.point(lambda p: 0)
    draw = ImageDraw.Draw(img)
    for key, value in mdict.items():
        for x, y in value:
            x /= scale
            y /= scale
            draw.ellipse((x - rad_dict[key], y - rad_dict[key], x + rad_dict[key], y + rad_dict[key]),
                         fill=200, outline=None)
    return img


def get_data_unet():
    img_width = 512
    img_height = 672
    annotations = glob('../annotations/*')

    images = np.ndarray(shape=(len(annotations), img_width,
                               img_height, 1), dtype=np.float32)
    masks = np.ndarray(shape=(len(annotations), img_width,
                              img_height, 1), dtype=np.float32)
    for i in range(len(annotations)):
        f, mdict = get_patch_coordinates(annotations[i])
        file_name = f.replace('tif', 'png')

        if os.path.isfile('../png_images/' + file_name):
            img = load_img('../png_images/' + file_name, grayscale=True,
                           target_size=(img_width, img_height))
            masked = get_masked_image(np.zeros((img_width, img_height), dtype=np.uint8), mdict,
                                      {1: 12, 2: 13, 4: 17, 5: 12})
            masked = np.array(masked) > 0
            masked_img = img * masked
            masked_img = np.expand_dims(
                masked_img, axis=0).transpose((1, 2, 0))

            images[i, :, :, :] = img_to_array(masked_img)

            masked = get_masked_image(np.zeros((img_width, img_height), dtype=np.uint8), mdict,
                                      {1: 2, 2: 2, 4: 2, 5: 2})
            masked = np.expand_dims(masked, axis=0).transpose((1, 2, 0))
            masks[i, :, :, :] = img_to_array(masked)
    return images, masks
