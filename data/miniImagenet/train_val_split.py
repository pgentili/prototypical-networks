import os
import random
from shutil import copyfile

# Splits each train class into 480/120 train/val for standard image
# classification training (i.e. 64 way many-shot)
data_dir = os.path.join(os.getcwd(), 'data')
split_path = os.path.join(os.getcwd(), 'splits', 'ravi', 'train')
train_dir = os.path.join(os.getcwd(), 'train_train')
val_dir = os.path.join(os.getcwd(), 'train_val')

with open('{}.txt'.format(split_path)) as class_file:
  for class_name in class_file:
    class_name = class_name.strip('\n')
    print(class_name)
    class_dir = os.path.join(data_dir, class_name)
    train_output_dir = os.path.join(train_dir, class_name)
    val_output_dir = os.path.join(val_dir, class_name)
    print(train_output_dir)
    for d in (train_output_dir, val_output_dir):
      if not os.path.exists(d):
        os.makedirs(d)

    class_images = os.listdir(class_dir)
    random.shuffle(class_images)
    num_images = len(class_images)
    num_train = int(0.8 * num_images)
    for image in class_images[:num_train]:
      copyfile(os.path.join(class_dir, image),
               os.path.join(train_output_dir, image))
    for image in class_images[num_train:]:
      copyfile(os.path.join(class_dir, image),
               os.path.join(val_output_dir, image))
