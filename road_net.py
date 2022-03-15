import tensorflow as tf

import tensorflow_hub as hub

import cv2
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

# For measuring the inference time.
import time

#Image
imgPath = 'F:\Forza Images\OffRoad5.png'

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

