from scipy.io import loadmat
from pprint import pprint as pp
import numpy as np
from matplotlib import pyplot as plt
from CarAnnotation import CarAnnotation
import os
from PIL import Image

# Method to parse a MAT file into a csv file
# mat_dir - path to the mat file
# output_file - output file name (i.e. output.txt)
# is_train - if the mat file is part of training, include parsing for the class
def mat_annotations_to_txt(mat_dir, output_file, is_train):

    if(os.path.exists(output_file)):
        print("annotations file already exists: " + output_file)
        return

    mat = loadmat(mat_dir)

    for k in mat.keys():
        pp([k, len(mat[k])])

    pp("Number of training examples: " + str(len(mat['annotations'][0])))
    pp(len(mat['annotations'][0][0]))

    with open(output_file, 'a') as f:
        for m in mat['annotations'][0]:

            x1 = m[0].item()
            x2 = m[1].item()
            y1 = m[2].item()
            y2 = m[3].item()

            if(is_train):
                classname = m[4].item()
                fname = m[5].item()
                f.write("{}, {}, {}, {}, {}, {}\n".format(x1, x2, y1, y2, classname,fname))
            else:
                fname = m[5].item()
                f.write("{}, {}, {}, {}, {}\n".format(x1, x2, y1, y2, fname))

# Parse through the given class names it the cars_meta.mat file
# Store the class names as a list in a text file
def classes_to_txt():

    #if(os.path.exists('class_names.txt')):
    #    return

    mat = loadmat('devkit/cars_meta.mat')

    for k in mat.keys():
        pp([k, len(mat[k])])

    print(len(mat['class_names'][0]))

    with open('class_names.txt', 'a') as f:
        for cn in mat['class_names'][0]:
            f.write("{}\n".format(cn[0]))

def image_sizes():

    images = os.listdir('./cars_train/')
    scale = 120, 80

    min_width = 100000
    min_height = 100000
    ratios = []

    fig = plt.figure()
    j = 1
    for i in images[0:9]:

        im = Image.open('./cars_train/Acura_Integra_Type_R_2001/'+ i)

        #im.thumbnail(scale)
        width, height = im.size  # Get dimensions
        if(width < min_width):
            min_width = width
        if(height < min_height):
            min_height = height

        im = im.resize(scale)
        ratios.append(width/height)

        plt.subplot(3, 3, j)
        plt.tight_layout()
        plt.imshow(im, cmap='gray', interpolation='none')
        j = j + 1

    plt.show()

    print("Average ratio: {}".format(sum(ratios)/len(ratios)))
    print("Min height: {}, min width: {}".format(min_height, min_width))

# Method to organize the training images into a format to be parsed by
# Torch's dataset.ImageFolder(...) function
def organize_training_images():
    with open('car_train_annotation.txt', 'r') as f:
        lines = f.readlines()

    class_files = []
    for line in lines:
        _, _, _, _, class_num, filename = line.replace(" ", "").strip().split(",")

        class_files.append([int(class_num), filename])

    for c in class_files[0:10]:
        pp(c)

    class_names = []
    with open('class_names.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        class_names.append(line.replace(" ", "_").replace("/", "_").strip())

    for c in class_names:
        dir = 'cars_train/' + c
        if not os.path.isdir(dir):
            os.mkdir(dir)

    for c in class_files:
        car_name = class_names[c[0] - 1]
        if os.path.exists('cars_train/' + c[1]):
            print('cars_train/' + car_name + "/" + c[1])
            new_dir = 'cars_train/' + car_name + "/" + c[1]
            os.rename('cars_train/'+c[1], new_dir, )

if __name__ == "__main__":
        pass
        #mat_annotations_to_txt()
        #classes_to_txt()
        #image_sizes()
        #organize_training_images()







