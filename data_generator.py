
from numpy.matlib import rand, random

f = 10 # frequency of states(frames per second)

#states = ['red', 'yellow', 'green', 'blinking_green', 'yellow_red']

color_map = {
    'red': (0, 0, 255),
    'yellow':(0, 255, 255),
    'green': (0, 255, 0),
    'yellow_red': (255, 0, 0)
}

def make_circle(data, f = 10):
    # Green
    f = 1
    green_time = random.randint(3, 6)
    for i in range(green_time * f):
        data.append(('green', 'green'))

    # Blinking green
    blinking_green_time = 7
    part_time = 0.45 + random.random() * 0.1
    time_tmp = 0
    color = 'green'
    while(time_tmp < blinking_green_time):
        data.append(('blinking_green', color))
        time_tmp = time_tmp + 1 / f
        if time_tmp > part_time:
            time_tmp = 0
            if color == 'black':
                color = 'green'
            else:
                color = 'black'
            blinking_green_time = blinking_green_time - part_time

    # Yellow
    yellow_time = 3
    for i in range(yellow_time * f):
        data.append(('yellow', 'yellow'))

    # Red
    red_time = random.randint(1, 20)
    for i in range(red_time * f):
        data.append(('red', 'red'))

    # Yellow_red
    red_time = 2
    for i in range(red_time * f):
        data.append(('yellow_red', 'yellow_red'))
    return data

def generate_data(f):
    data = []
    for i in range(15):
        make_circle(data, f)
    #blank_image = np.zeros((800, 600, 3), np.uint8)
    #for o in data:
        #cv2.circle(blank_image, (300,300), 300, color_map.get(o[1]), thickness=-1)
       # imshow('colors', blank_image)
        #cv2.waitKey(1)
        #time.sleep(1/f)
    print(data)
    return data