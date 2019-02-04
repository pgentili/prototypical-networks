import cv2
import os
from shutil import copyfile

splits = ['train', 'val', 'test']

for split in splits:
  split_dir = os.path.join(os.getcwd(), split)
  if not os.path.exists(split_dir):
    os.makedirs(split_dir)

  classes = set()
  with open('{}.csv'.format(split)) as split_file:
    for line in split_file:
      if line[0] == 'n':
        class_name = line[:-1].split(',')[1]
        classes.add(class_name)

  with open('{}.txt'.format(split), 'w') as class_file:
    for class_name in classes:
      class_dir = os.path.join(split_dir, class_name)
      if not os.path.exists(class_dir):
        os.makedirs(class_dir)
      class_file.write('{}\n'.format(class_name)) 

  #image_dir = os.path.join(os.getcwd(), 'images')
  #for image_name in os.listdir(image_dir):
  #  class_name = image_name[:9]
  #  if class_name in classes:
  #    #im = cv2.imread(os.path.join(image_dir, image_name))
  #    #im_resized = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)
  #    #cv2.imwrite(os.path.join(split_dir, class_name, image_name), im_resized)
  #    copyfile(os.path.join(image_dir, image_name),
  #             os.path.join(split_dir, class_name, image_name))
               
