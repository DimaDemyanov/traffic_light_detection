import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from os import path
from traffic_light_detector.utils import label_map_util
from traffic_light_detector.utils import visualization_utils as vis_util
import time
import cv2

i = 0

def detect(img, lower, upper, Threshold=0.05):
    """
        detect red and yellow
        :param img:
        :param Threshold:
        :return:
        """

    desired_dim = (30, 90)  # width, height
    img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lower_red = np.array([lower,25,240])
    upper_red = np.array([upper,50,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([lower, 70, 150])
    upper_red = np.array([upper, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # cv2.imshow('mask0',mask0)
    # cv2.imshow('mask1',mask1)

    # red pixels' mask
    mask = mask0+mask1

    cv2.imshow('mask_red' + str(lower), mask)

    # Compare the percentage of red values
    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

    if rate > Threshold:
        return True
    else:
        return False


def detect_red(img, Threshold=0.05):
    return detect(img, 120, 145, Threshold)


def detect_yellow(img, Threshold=0.05):
    return detect(img, 90, 110, Threshold)


def detect_green(img, Threshold=0.05):
    return detect(img, 30, 60, Threshold)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)




def read_traffic_lights(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.1, traffic_ligth_label=10):
    im_width, im_height = image.size
    red_flag = False
    crop_img = None

    mleft, mright, mtop, mbottom = 0, 0, 0, 0
    n = 0
    (left, right, top, bottom) = None, None, None, None
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if n == 1:
            break
        if scores[i] > min_score_thresh and classes[i] == traffic_ligth_label:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            ## print(left, right, top, bottom)
            mleft = mleft + left
            mright = mright + right
            mtop = top + mtop
            mbottom = mbottom + bottom
            n = n + 1

            # crop_img = image.crop((left, top, right, bottom))
            # i = 0

            if crop_img:
                print(detect_red(crop_img,0.1))
                img = load_image_into_numpy_array(crop_img)
                ## print(img.shape)
                img = cv2.resize(img, (300, 500))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow('wow' + str(i), img)
                i = i + 1

    #time.sleep(0.3)

    if n != 0:
        mleft = mleft / n
        mright = mright / n
        mtop = mtop / n
        mbottom = mbottom / n
    return (mleft, mright, mtop, mbottom)


def plot_origin_image(image_np, boxes, classes, scores, category_index):

    # Size of the output images.
    IMAGE_SIZE = (12, 8)
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      min_score_thresh=.5,
      use_normalized_coordinates=True,
      line_thickness=3)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

    # save augmented images into hard drive
    # plt.savefig( 'output_images/ouput_' + str(idx) +'.png')
    plt.show()


def detect_traffic_lights(MODEL_NAME, plot_flag=False):
    """
    Detect traffic lights and draw bounding boxes around the traffic lights
    :param PATH_TO_TEST_IMAGES_DIR: testing image directory
    :param MODEL_NAME: name of the model used in the task
    :return: commands: True: go, False: stop
    """

    #--------test images------
    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}image.bmp'.format(i)) for i in range(0, Num_images) ]


    commands = []

    # What model to download
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

    # number of classes for COCO dataset
    NUM_CLASSES = 90


    #--------Download model----------
    if path.isdir(MODEL_NAME) is False:
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    #--------Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    #----------Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)




    return detection_graph

if __name__ == "__main__":
    Num_images = 17
    #PATH_TO_TEST_VIDEO = './videos/3.MOV'
    PATH_TO_TEST_VIDEO = './videos/2.mp4'
    MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'

    commands = detect_traffic_lights(PATH_TO_TEST_VIDEO, MODEL_NAME, Num_images, plot_flag=False)
    print(commands)
