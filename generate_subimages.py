import xml.etree.ElementTree as ET
from skimage.io import imread, imsave
import numpy as np
import pathlib

def parseBndbox(element):
    xmin = element.find('xmin').text
    xmax = element.find('xmax').text
    ymin = element.find('ymin').text
    ymax = element.find('ymax').text

    return [int(xmin), int(xmax), int(ymin), int(ymax)]

dir = "VOC2007_train/"
files = pathlib.Path(dir)
files = list(files.glob('Annotations/*'))
files = [str(f) for f in files]
#print(files)

for f in files:
    label = ET.parse(f)
    objs = label.findall('object')
    jpg_filename = label.find('filename').text
    #print(jpg_filename)

    img = imread(dir+"/JPEGImages/"+jpg_filename)

    for obj in objs:
        truncated = int(obj.find('truncated').text)
        difficult = int(obj.find('difficult').text)
        #print(str(truncated) + ', ' + str(difficult))

        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        #print(name)
        #print(bndbox)
        # Filter out difficult, occluded samples
        if not truncated or not difficult:
            [xmin, xmax, ymin, ymax] = parseBndbox(bndbox)
            # Filter out images that are too small
            if (xmax-xmin)*(ymax-ymin) > 9000:
                img_index = jpg_filename.split('.')[0]
                #print(img_index)
                cropped_img = np.copy(img[ymin:ymax, xmin:xmax])
                imsave(dir+img_index+'_'+name+'.jpg', cropped_img)
