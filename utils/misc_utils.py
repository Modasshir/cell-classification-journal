# coding: utf-8

# In[1]:

import itertools
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

import sys

if sys.version_info[0] < 3:
    from skimage import filters as filter
else:
    from skimage import filters as filter


# In[2]:


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          Plot=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if Plot:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    if Plot:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    return cm


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


def get_segmented_cells(im, do_clear=False, plot=False):
    '''
    This function segments the cells from input whole image.
    '''
    if plot == True:
        f, plots = plt.subplots(9, 1, figsize=(5, 45))

    '''
    Step 0: Equalize histogram of the original image
    '''
    #     im=skimage.exposure.rescale_intensity(im.astype(np.uint8))
    #     tmp = skimage.exposure.equalize_adapthist(im.astype(np.uint8),nbins=256)*255
    tmp = im.copy()
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(tmp.astype(np.uint8), cmap='gray')
        plots[0].set_title('histogram equalization applied')

    '''
    Step 1: Convert into a binary image. 
    '''
    binary = tmp > filters.threshold_otsu(tmp)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(binary, cmap='gray')
        plots[1].set_title('thresholded binary image')

    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    if do_clear:
        cleared = clear_border(binary)
    else:
        cleared = binary.copy()

    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(cleared, cmap='gray')
        plots[2].set_title('Border Cleared')
    '''
    Step 3: Label the image.
    '''

    label_image = label(cleared)

    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(label_image, cmap='gray')
        plots[3].set_title('Labeled Image')

    '''
    Step 4: remove the labels with smaller area than the threshold.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    for region in regionprops(label_image):
        if region.area < 5:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0

    binary = label_image > 0

    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(label_image, cmap='gray')
        plots[4].set_title('Smallest cells shapes removed')

    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''

    #     disk_size=1
    #     selem = disk(disk_size)
    #     binary = binary_erosion(binary, selem)
    #     if plot == True:
    #         plots[5].axis('off')
    #         plots[5].imshow(binary, cmap='gray')
    #         plots[5].set_title('binary erosion with disk size '+str(disk_size))
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''

    #     disk_size=3
    #     selem = disk(disk_size)
    #     binary = binary_closing(binary, selem)

    #     if plot == True:
    #         plots[6].axis('off')
    #         plots[6].imshow(binary, cmap='gray')
    #         plots[6].set_title('binary closing with disk size '+str(disk_size))
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = filters.roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(binary, cmap='gray')
        plots[7].set_title('small holes filled')

    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    #     binary = binary_opening(binary, selem)

    get_low_vals = binary == 0
    mask = im.copy()
    mask[get_low_vals] = 0

    if plot == True:
        plots[8].axis('off')
        plots[8].imshow(mask, cmap='gray')
        plots[8].set_title('mask applied')

    return mask.astype(np.uint8)


def plot(img):
    plt.figure(figsize=(15, 15))
    plt.imshow(img, cmap='gray')
    plt.show()


def get_annotated_image(file_name, mdict, radius=3):
    if type(file_name) == str:
        img = Image.open('../png_images/' + file_name, mode='r')
    else:
        img = Image.fromarray(file_name)
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("arial.ttf", 15)

    for key, value in mdict.items():
        for x, y in value:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=200, outline=None)
            draw.text((x, y), str(key), font=fnt, fill=200)
    return img
