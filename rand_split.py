
import random
import os

def random_split(main_folder):
    sub_folders = os.listdir(main_folder)

    for sf in sub_folders:
        sub = main_folder + '/' + sf
        pics = os.listdir(sub)
        print(sub + ": " + str(len(pics)))

if __name__ == '__main__':
    random_split('cars_train')