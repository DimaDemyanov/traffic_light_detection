####To install run on Windows with Anaconda

conda env create traffic-light-detector/environment-gpu.yml
 
conda install tensorflow=1.13.1

conda install tensorflow-gpu

conda install -c menpo opencv3

####Then run

python main.py 

#### How does it work??

You can find test videos in directory ./videos
After running ./main.py you will see 5 frames
First - image with only traffic light (algorithm that finds traffic lights is Faster RCNN trained on COCO 
data set on 11.06.2017)

More three frames are traffic light masks made with threshold filter in HSV. 
So three for green, red and yellow colors

And last frame shows given video frame with two traffic lights. Right one shows detected colors made with 
threshold filter. And left one shows specified traffic light state.

The result videos by default will be in ./videos_res

You can choose your videos to be handled by changing 

PATH_TO_TEST_VIDEO_DIR, PATH_TO_TEST_VIDEOS in main.py

