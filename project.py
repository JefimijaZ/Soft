import numpy as np
import image as im
import cv2
import os
# import knn as kn

from scipy import ndimage
from keras.models import load_model
from vector import distance, pnt2line
from matplotlib.pyplot import cm
import itertools
import time

model = load_model('model.h5')


def get_hough_line(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grey, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=100)
    # print lines
    xmin, ymin, xmax, ymax = [], [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            xmin.append(x1)
            ymin.append(y2)
            xmax.append(x2)
            ymax.append(y1)
    xmin1, ymin1, xmax1, ymax1 = min(xmin), min(ymin), max(xmax), max(ymax)
    cv2.line(grey, (xmin1, ymax1), (xmax1, ymin1), (255, 0, 0), 2)
    return xmin1, ymax1, xmax1, ymin1


def inRange(r, item, items):
    # print item
    retVal = []
    for obj in items:
        if (distance(item['center'], obj['center']) < r):
            retVal.append(obj)
    return retVal


def prepare_predict(frm):
    grey = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    retVal = im.prepare_for_ann(grey)
    retVal = cv2.resize(retVal, (28, 28), interpolation=cv2.INTER_LINEAR)
    retVal = model.predict_classes(retVal.reshape(1, 1, 28, 28).astype('float32'))
    return retVal


idNext = -1


def nextId():
    global idNext
    idNext += 1
    return idNext





def read_cap(path):
    cap = cv2.VideoCapture(path)
    return cap

file = open('video/out.txt','w')
for i in range(0, 10):
    t = 0
    counter = 0
    suma = []
    passed_elements2 = []
    elements = []
    times = []
    name = 'video/video-' + str(i) + '.avi'
    path = os.path.join(os.getcwd(), name)
    cap = read_cap(path)
    ret, frame = cap.read()
    blue = im.in_range(frame, [0, 0, 180], [50, 50, 255])
    xmin1, ymax1, xmax1, ymin1 = get_hough_line(blue)
    bline = [(xmin1, ymax1), (xmax1, ymin1)]
    # cv2.line(frame, (xmin1, ymax1), (xmax1, ymin1), (0, 0, 255), 2)
    green = im.in_range(frame, [0, 180, 0], [50, 255, 50])
    xmin2, ymax2, xmax2, ymin2 = get_hough_line(green)
    gline = [(xmin2, ymax2), (xmax2, ymin2)]

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break;

        # cv2.line(frame, (xmin2, ymax2), (xmax2, ymin2), (0, 0, 255), 2)
        # f = im.select_roi(frame)
        frame1 = frame
        lower = np.array([230, 230, 230])
        upper = np.array([255, 255, 255])
        image_track = im.in_range(frame1, lower, upper)  # samo brojevi
        image_track = im.dilate_image(image_track)
        lab, objects1 = ndimage.label(image_track)
        objects = ndimage.find_objects(lab)
        for i in range(objects1):
            loc = objects[i]
            xc = int((loc[1].stop + loc[1].start) / 2)
            yc = int((loc[0].stop + loc[0].start) / 2)
            (xc, yc) = (xc, yc)
            dxc = int(loc[1].stop - loc[1].start)
            dyc = int(loc[0].stop - loc[0].start)
            (dxc, dyc) = (dxc, dyc)
            # print xc, yc,dxc, dyc
            if (dxc > 11 or dyc > 11):
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                # print t
                (x, y) = (loc[1].start, loc[0].start)
                lst = inRange(20, elem, elements)  # ako su dovoljno daleko
                nn = len(lst)
                if nn == 1:
                    value = prepare_predict(frame1[y: y + dyc, x: x + dxc])
                    lst[0]['value'] = value
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t, 'val': value})
                    lst[0]['future'] = []
                elif nn == 0:
                    elem['id'] = nextId()
                    elem['passB'] = False
                    elem['passG'] = False
                    elem['t'] = t
                    value = prepare_predict(frame1[y: y + dyc, x: x + dxc])
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t, 'val': value}]
                    elem['value'] = value
                    elem['future'] = []
                    elements.append(elem)

        for el in elements:
            tt = t - el['t']
            if (tt < 3):
                dist, pnt, r = pnt2line(el['center'], bline[0], bline[1])
                c = (25, 25, 255)
                if r > 0 and dist < 9:
                    c = (0, 255, 160)
                    if el['passB'] == False:
                        el['passB'] = True
                        # if (dist > 9 & el['passB'] == True):
                        c = (0, 255, 255)
                        valH = [i['val'] for i in el['history']]
                        h = [i[0] for i in valH]
                        counter += np.argmax(np.bincount(h))

                cv2.circle(frame1, el['center'], 16, c, 2)
                dist, pnt, r = pnt2line(el['center'], gline[0], gline[1])
                c = (25, 25, 255)
                if r > 0 and dist < 9:
                    c = (0, 255, 160)
                    if el['passG'] == False:
                        el['passG'] = True
                        valH = [i['val'] for i in el['history']]
                        h = [i[0] for i in valH if i != -1]
                        counter -= np.argmax(np.bincount(h))

                cv2.circle(frame1, el['center'], 16, c, 2)  # debeli crveni krug


        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        # print counter
        cv2.putText(frame1, 'SUMA: ' + str(counter), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        # print nr_objects
        t += 1
        # if t % 10 == 0:
        # print t
        cv2.line(frame1, (xmin1, ymax1), (xmax1, ymin1), (0, 0, 255), 2)
        cv2.line(frame1, (xmin2, ymax2), (xmax2, ymin2), (0, 0, 255), 2)
        cv2.imshow('frame', frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    file.write('video- '+str(i)+'.avi \t' + str(counter) + '\n')
    print 'suma = ' + str(counter)

cap.release()
cv2.destroyAllWindows()
file.close()
