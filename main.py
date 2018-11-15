import cv2
from matplotlib import pyplot as plt
import numpy as np


class Circle:
    def __init__(self, cont, x, y):
        self.x = x
        self.y = y
        self.cont = cont


class Animal:
    def __init__(self, pic, color):
        self.pic = pic
        self.animals = []
        self.color = color
        self.center = None


def find_circle(c):
    x, y, w, h = cv2.boundingRect(c)
    crop = im[y:y + h, x:x + w]
    return Circle(crop, x, y)


def find_center_animals(center_circle, animals):
    center_animals = []
    for animal in animals:
        animal_coords = find_animal(center_circle, animal)
        if animal_coords is not None:
            animal.center = animal_coords
            center_animals.append(animal)
    return center_animals


def find_animal(circ, animal):
    circle = circ.cont
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread(animal.pic, 0)
    img2 = circle

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        rectX = 0
        rectY = 0
        for i in dst:
            rectX += i[0][0]
            rectY += i[0][1]
        rectX /= 4
        rectY /= 4
        # for i in animal.animals:
        #     cv2.line(im, (int(rectX+circ.x), int(rectY+circ.y)), (i[0], i[1]), animal.color, 5)
        # animal.animals.append([int(rectX + circ.x), int(rectY + circ.y)])
        return (int(rectX + circ.x), int(rectY + circ.y))
    return None


im = cv2.imread('pics/1.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(imgray, 192, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
height = im.shape[0]
width = im.shape[1]
center = [width / 2, height / 2]
center_contour = contours[0]
mindistance = abs(center[1] - center_contour[0][0][1]) + abs(center[0] - center_contour[0][0][0])
contours = list(filter(lambda x: cv2.contourArea(x) >= 100, contours))

position = 0
for i, con in enumerate(contours):
    c = con[0][0]
    distance = abs(center[1] - c[1]) + abs(center[0] - c[0])
    if distance < mindistance:
        center_contour = con
        mindistance = distance
        position = i
# cv2.drawContours(im, contours, -1, (0, 0, 255), 20)

center_circle = find_circle(center_contour)
del contours[position]
circles = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    crop = im[y:y + h, x:x + w]
    circle = Circle(crop, x, y)
    circles.append(circle)

animals = [Animal('pics/osiol.png', (255, 0, 0)),
           Animal('pics/mewa.png', (0,255,0)),
           Animal('pics/delfin.png', (0, 0, 255)),
           Animal('pics/flaming.png', (120,255,0)),
           Animal('pics/foka.png', (50, 0, 110)),
           Animal('pics/pingwin.png', (110, 50, 0)),
           Animal('pics/wiewiorka.png', (255,0,255)),
           Animal('pics/zaba.png', (0,255,255)),
           Animal('pics/zolw.png', (255,255,0))]

center_animals = find_center_animals(center_circle, animals)
for animal in center_animals:
    for circ in circles:
        coords = find_animal(circ, animal)
        if coords is not None:
            animal.animals.append(coords)
    for i in animal.animals:
        cv2.line(im, animal.center, (i[0], i[1]), animal.color, 5)
plt.imshow(im)
plt.show()
