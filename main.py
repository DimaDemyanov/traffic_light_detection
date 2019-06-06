import numpy as np
from collections import Counter
import cv2
from PIL import Image
import hmm_my
import traffic_light_detector
from hmm_my import get_model
from state_detection import draw_traffic_light
from traffic_light_detector.main import load_image_into_numpy_array, detect_traffic_lights, read_traffic_lights, \
    detect_red, detect_yellow, detect_green
import tensorflow as tf


if __name__ == "__main__":
    MAX_CNT_HMM_STATES = 1000
    MAX_CNT_STATES_TO_SPECIFY = 10
    CNT_FRAMES_USING_SAME_TLPOS = 5
    PATH_TO_RESULT_VIDEO_DIR = './videos_res/'
    PATH_TO_TEST_VIDEO_DIR = './videos/'
    # PATH_TO_TEST_VIDEO = '2.mp4'
    # PATH_TO_TEST_VIDEO = '11_1.mp4'
    # PATH_TO_TEST_VIDEO = '10.mp4'
    # PATH_TO_TEST_VIDEO = '13.mp4'
    # PATH_TO_TEST_VIDEO = '14.MOV'
    # PATH_TO_TEST_VIDEO = '9.MOV'
    PATH_TO_TEST_VIDEOS = ['2.mp4','11_1.mp4', '10.mp4', '14.MOV', '9.MOV', '16.mp4', '18.MOV']
    # PATH_TO_TEST_VIDEOS = ['2.mp4']

    MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'

    for PATH_TO_TEST_VIDEO in PATH_TO_TEST_VIDEOS:
        detection_graph = detect_traffic_lights(MODEL_NAME, plot_flag=False)

        left, top, right, bottom = 0, 0, 10, 10
        cnt = 0

        colors_n = []

        # Opening test video file
        cap = cv2.VideoCapture(PATH_TO_TEST_VIDEO_DIR + PATH_TO_TEST_VIDEO)
        if not cap.isOpened():
            cap = cv2.VideoCapture(PATH_TO_TEST_VIDEO_DIR + PATH_TO_TEST_VIDEO)
            cv2.waitKey(1000)
            print("Wait for the header")
            exit(1)

        # Getting video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

        # Opening output video file with same dimensions as opened test file and 25 fps
        out = cv2.VideoWriter(PATH_TO_RESULT_VIDEO_DIR + PATH_TO_TEST_VIDEO, 0x7634706d, 25,
                              (width, height))
        hmm = get_model(int(cap.get(cv2.CAP_PROP_FPS)), hmm_my.Data.DEFAULT_DATA)

        with tf.device('/device:GPU:0'):
            with detection_graph.as_default():
                # Opening tensorflow session
                with tf.Session(graph=detection_graph) as sess:
                    # Definite input and output Tensors for detection_graph
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                i = 0

                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        break

                    if cv2.waitKey(10) == ord('s'):
                        print('saving image')
                        cv2.imwrite('images/' + str(i) + 'image.bmp', frame)
                        i = i + 1

                    if cv2.waitKey(10) == ord('w'):
                        print('breaking')
                        out.release()
                        break

                    # Converting frame to Pillow.Image format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)

                    # Find traffic light position on the frame
                    if cnt % CNT_FRAMES_USING_SAME_TLPOS == 0:
                        # the array based representation of the image will be used later in order to prepare the
                        # result image with boxes and labels on it.
                        image_np = load_image_into_numpy_array(image)

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)

                        # Actual detection.
                        (boxes, scores, classes, num) = sess.run(
                          [detection_boxes, detection_scores, detection_classes, num_detections],
                          feed_dict={image_tensor: image_np_expanded})

                        (nleft, nright, ntop, nbottom) = read_traffic_lights(image, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))

                        if nleft:
                            left, top, right, bottom = nleft, ntop, nright, nbottom

                    crop_img = image.crop((left, top, right, bottom))

                    # Show traffic light image
                    if crop_img and left != 0:
                        img = load_image_into_numpy_array(crop_img)
                        img = cv2.resize(img, (300, 500))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow('crop_tl', img)


                    # Detecting active colors
                    colors = []

                    if(detect_red(crop_img)):
                        colors.append('red')

                    if (detect_yellow(crop_img)):
                        colors.append('yellow')

                    if (detect_green(crop_img)):
                         colors.append('green')

                    # Drawing active colors on the frame(right side of the frame)
                    draw_traffic_light(frame, colors, frame.shape[1] - 45, 0)

                    # Detecting state basing on active colors
                    if 'green' in colors:
                        colors_n.append('green')
                    else:
                        if 'yellow' in colors and 'red' in colors:
                            colors_n.append('yellow_red')
                        else:
                            if 'yellow' in colors:
                                colors_n.append('yellow')
                            else:
                                if 'red' in colors:
                                    colors_n.append('red')
                                else:
                                    colors_n.append('black')

                    # Specifing traffic light state using HMM model
                    y = np.array([hmm_my.colors.index(o) for o in colors_n])
                    _, x = hmm.decode(np.array([hmm_my.colors.index(o) for o in colors_n]).reshape(-1, 1))
                    count = Counter(x[-MAX_CNT_STATES_TO_SPECIFY:])
                    count.most_common()

                    draw_traffic_light(frame, hmm_my.state_color[hmm_my.states[count.most_common()[0][0]]], 0, 0)

                    if len(colors_n) > MAX_CNT_HMM_STATES:
                        colors_n.pop(0)
                    # print(y)
                    # print(x)

                    out.write(frame)
                    # Showing result frame with traffic lights
                    cv2.imshow('result', frame)
                    cnt = cnt + 1
        out.release()