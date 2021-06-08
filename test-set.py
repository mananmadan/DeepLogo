#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import argparse

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', required=True, help='Path to model')
  parser.add_argument('--label_map', required=True, help='Path to label map text')
  parser.add_argument('--test_annot_text', required=True, help='Path to test annotation text')
  parser.add_argument('--test_image_dir', required=True, help='Path to test image directory')
  parser.add_argument('--output_dir', required=True, help='Path to output directory')
  return parser.parse_args()

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

if __name__ == "__main__":
  args = parse_arguments()
  MODEL_NAME = args.model_name

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = args.label_map

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

  data = np.loadtxt(args.test_annot_text, delimiter=',', dtype=str)
  print(data)
  cat = data[:,-1]
  test_set_images = data[:, 0]
  print(test_set_images)
  print("cat data:",cat)
  print(len(cat))
  print(len(test_set_images))

  PATH_TO_TEST_IMAGES_DIR = args.test_image_dir
  TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, img) for img in test_set_images]

  # Size, in inches, of the output images.
  IMAGE_SIZE = (12, 8)

  if not os.path.exists(os.path.abspath(args.output_dir)):
    os.makedirs(args.output_dir)
  print(category_index[18])
  print(category_index[19])
  cnt = 0
  correct = 0
  FP = 0
  FN = 0
  
  for i in category_index:
      print(i,category_index[i])
  correct_dict = {}
  wrong_dict = {}
  for i, image_path in enumerate(TEST_IMAGE_PATHS):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
    detect = 0
    correct_detect = 0
    print("desired class",category_index[int(cat[cnt])+1]['name'])
    desired_class = category_index[int(cat[cnt])+1]['name']
    for j in range(0,len(output_dict['detection_classes'])):
        if output_dict['detection_scores'][j] > 0.5:
            detect = 1
            detected_class = category_index[output_dict['detection_classes'][j]]['name']
            print("detected_class",detected_class)
            #print("checking cat index:",int(cat[cnt])+1)
            if detected_class == category_index[int(cat[cnt])+1]['name']:
                if desired_class not in correct_dict:
                    correct_dict[desired_class] = 1
                else:
                    correct_dict[desired_class] = correct_dict[desired_class]+1
                correct_detect = 1
                correct = correct + 1
                break
    if detect == 1 and correct_detect == 0:
        if desired_class not in wrong_dict:
            wrong_dict[desired_class] = 1
        else:
            wrong_dict[desired_class] = wrong_dict[desired_class]+1
        FP = FP + 1
    if detect == 0:
        FN = FN + 1
    print("done cnt:",cnt,"correct count",correct)
    cnt = cnt + 1
    print("correct percentage:",correct/len(cat))
    print("false negative:",FN)
    print("false positive:",FP)

print("wrong_dict")
for i in wrong_dict:
    print(i,wrong_dict[i])
print("correct_dict")
for i in correct_dict:
    print(i,correct_dict[i])
