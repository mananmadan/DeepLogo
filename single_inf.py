
import numpy as np
import os
import sys
import tarfile
import math
import tensorflow.compat.v1 as tf
import zipfile
import cv2
from google.colab.patches import cv2_imshow
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw
try:
  from object_detection.utils import label_map_util
  from object_detection.utils import visualization_utils as vis_util
  from object_detection.utils import ops as utils_ops
except:
  from DeepLogo.object_detection.utils import label_map_util
  from DeepLogo.object_detection.utils import visualization_utils as vis_util
  from DeepLogo.object_detection.utils import ops as utils_ops
from numpy import asarray


##new load function
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    if image.getdata().mode != "RGB":
        image = image.convert('RGB')
    #print(image.getdata().mode)
    np_array = np.array(image.getdata())
    reshaped = np_array.reshape((im_height, im_width, 3))
    return reshaped.astype(np.uint8)
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

def main(model_name,label_map,path):
  MODEL_NAME = model_name
  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = label_map
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)
  # Size, in inches, of the output images.
  IMAGE_SIZE = (12, 8)
  file_dict = {}
  image_path = str(path)
    #print("opening image ..",image_path)
  image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
  mx = 0
  detected = ""
  image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
  cnt = 0
  for j in output_dict['detection_classes']:
    if output_dict['detection_scores'][cnt] > mx:
        mx = output_dict['detection_scores'][cnt]
        detected = category_index[j]['name']
    cnt = cnt + 1
  return (detected,mx)
 
def pre(model_name,label_map):
  MODEL_NAME = model_name
  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = label_map
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)
  return (category_index,detection_graph)
  # Size, in inches, of the output images.

def main_img(category_index,detection_graph,img):
  cv2.imwrite("temp.jpg",img)
  image_path = "temp.jpg"
  try:
    image = Image.open(image_path)
    mx = 0
    detected = ""
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
    cnt = 0
    for j in output_dict['detection_classes']:
      if output_dict['detection_scores'][cnt] > mx:
          mx = output_dict['detection_scores'][cnt]
          detected = category_index[j]['name']
      cnt = cnt + 1
    return (detected,mx)
  except:
    return ("",-1)

def getallbox(category_index,detection_graph,img):
  cv2.imwrite("temp.jpg",img)
  image_path = "temp.jpg"
  try:
    image = Image.open(image_path)
    mx = 0
    detected = ""
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
    print("New")
    (im_width, im_height) = image.size
    cnt = 0
    print(img.size)
    for j in output_dict['detection_classes']:
      print("boxes",output_dict['detection_boxes'][cnt],"scores",output_dict['detection_scores'][cnt],"name",category_index[j]['name'])
      bb = [output_dict['detection_boxes'][cnt]]
      ymin = bb[0][0]
      xmin = bb[0][1]
      ymax = bb[0][2]
      xmax = bb[0][3]
      (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
      left = math.floor(left)
      right = math.ceil(right)
      top = math.floor(top)
      bottom = math.ceil(bottom)

      print(left,right,top,bottom)
      ## segemented image
      timg = img[top:bottom,left:right,:]
      cv2.imwrite(str(cnt)+".jpg",timg)
      if output_dict['detection_scores'][cnt] > mx:
          mx = output_dict['detection_scores'][cnt]
          detected = category_index[j]['name']
      cnt = cnt + 1
    return (detected,mx)
  except:
    return ("",-1)
