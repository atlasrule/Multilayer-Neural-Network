# Reads train and test image data

from PIL import Image
import glob
from warnings import warn

image_size = (28, 28)

def train_outputs(digit_label):
  expected_output = [0 for i in range(10)]
  expected_output[digit_label] = 1
  return expected_output

class LearningData:
  inputs: list
  outputs: list

def get_data(data_type):
  valid_types = {'train', 'test'}
  if data_type not in valid_types:
    raise ValueError("get_data: data_type must be train or test.")

  images_data = []
  images_labels = []
    
  for label in range(10):
    image_paths = glob.glob('mnist-{}/{}/*.png'.format(data_type, label))
    

    for image_path in image_paths:
      image = Image.open(image_path, 'r')
      
      width, height = image.size
      if width != image_size[0] or height != image_size[1]:
        warn("get_data: incompatible image size for:", image_path)
  
      images_data.append(list(image.getdata()))
      images_labels.append(train_outputs(label))

  learning_data = LearningData()
      
  learning_data.inputs = images_data
  learning_data.outputs = images_labels
  
  return learning_data
