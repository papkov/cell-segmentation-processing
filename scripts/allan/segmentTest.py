import numpy as np
import cv2
from scipy import ndimage

from skimage.feature import peak_local_max
from skimage.morphology import watershed

import skimage

import imutils

from itertools import cycle

import sys

from itertools import chain

from shapely.geometry import Polygon, LineString

import random as r

import csv
import time

#data = np.load('data/fluorescent.npy')
fileName = "HepG2"
data = np.load('data/' + fileName + '.npz')

filePrefix = "res_" + fileName + "_split/t"

csvFile = "cellCount-" + fileName + ".csv"

dataIndexes = list(range(0, 431))
logCSV = False
#dataIndexes = [24, 35, 39, 51, 55, 56, 66, 68, 71, 72, 84, 98, 164, 213, 224, 228, 253, 301, 330, 333, 337, 349, 370, 382, 385, 399, 408, 411, 421]

dataIndexes = [0]

print(data.files)

segData = data['bf_seg']

print(segData.shape)


def concaveExtraction(points, contours):
    cntLen = len(contours)

    if cntLen < 3:
        return points

    pool = cycle(contours)
    counter = 0
    pre = cur = nex = None

    concaveAngles = []

    preList = []
    curList = []
    nexList = []

    while counter < cntLen:
        if pre is None:
            pre = np.array(next(pool))
            cur = np.array(next(pool))
            nex = np.array(next(pool))
        else:
            pre = cur
            cur = nex
            nex = np.array(next(pool))

        aPre = np.arctan2(pre[1] - cur[1], pre[0] - cur[0])
        aNex = np.arctan2(nex[1] - cur[1], nex[0] - cur[0])

        conc = np.abs(aPre - aNex)

        if conc < np.pi:
            conc = conc
        else:
            conc = np.pi - conc

        # print(np.rad2deg(conc))

        preList.append((pre[0], pre[1]))
        curList.append((cur[0], cur[1]))
        nexList.append((nex[0], nex[1]))

        container = {"angel": conc, "pre": pre, "cur": cur, "nex": nex}

        concaveAngles.append(container)
        counter = counter + 1

    concaveAngles.insert(0, concaveAngles.pop())

    # print(preList)
    # print(curList)
    # print(nexList)
    # print("----")
    # sys.exit()
    return concaveAngles


def convertContours(cnts):
    cntList = []
    for cnt in cnts:
        cntList.append(cnt[0])
    return cntList


def takeLargerIfDiffSmall(list, minDiff):

    if len(list) < 2:
        return list

    print(list)

    pool = cycle(list)
    loopLimit = len(list)

    newList = []

    prev = None
    #todoFirst = True
    for i in range(loopLimit):
        if prev is None:
            prev = next(pool)
            continue
        cur = next(pool)

        print(newList, cur, prev)

        if (cur - prev) > minDiff:
            newList.append(prev)
            newList.append(cur)
        #elif todoFirst:
        #    todoFirst = False
        #    newList.append(cur)
        else:
        #    newList.pop()
        #    newList.append(cur)
            newList.append(cur)
        prev = cur

    print("NEW LIST:", newList)

    return newList


def sliceList(list, indexes):

    print("INDEXES:", indexes)

    sortedIndexes = sorted(indexes)

    sortedIndexes = takeLargerIfDiffSmall(sortedIndexes, 3)

    if len(sortedIndexes) < 1:
        return [list]

    smallestIndex = sortedIndexes[0]
    correctedList = list[smallestIndex:] + list[:smallestIndex]

    print(list)
    print(correctedList)

    slicedList = []
    previousIndex = -1
    for index, obj in enumerate(sortedIndexes):
        if index == 0:
            previousIndex = obj - smallestIndex
            continue

        correctedIndex = obj - smallestIndex

        print("taking indexes:", previousIndex, correctedIndex)

        if previousIndex == 0:
            slicedList.append([correctedList[-1]] + correctedList[previousIndex: correctedIndex + 1])
        else:
            slicedList.append(correctedList[previousIndex: correctedIndex + 1])
        previousIndex = correctedIndex
    slicedList.append(correctedList[sortedIndexes[-1] - smallestIndex:])

    return slicedList


def calcMiddlePoint(p1, p2):
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


def mergeArrays(arr1, arr2):
    newArr = []

    for el in arr1:
        newArr.append(el)
        if len(arr2) > 0:
            newArr.append(arr2.pop(0))

    if len(arr2) > 0:
        newArr.append(arr2)

    return newArr


def linearExpandLine(points, minLenth):

    print("linear expand", points, minLenth)

    if len(points) >= minLenth or len(points) < 2:
        return points

    tooShort = True

    mergedList = points.copy()

    while tooShort:

        newPoints = []

        prevPnt = None
        for index, obj in enumerate(mergedList):
            if prevPnt is None:
                prevPnt = obj
                continue

            newPoint = calcMiddlePoint(prevPnt, obj)
            newPoints.append(newPoint)
            prevPnt = obj

        print("merged vs new points", mergedList, newPoints)
        #mergedList = list(chain(*zip(mergedList, newPoints)))
        mergedList = mergeArrays(mergedList, newPoints)
        print("after merge", mergedList)

        if len(mergedList) >= minLenth:
            tooShort = False

    return mergedList


def lineInContour(cnt, line):
    try:
        polygon = Polygon(cnt)
        line = LineString(line)
        return not line.touches(polygon)
    except:
        return False


def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((int(point[0]), int(point[1])))
    return ret


def evaluateCavityForMirror(cavity):
    if np.sign(cavity) < 0:
        inverseCavity = np.pi - np.abs(cavity)
    else:
        inverseCavity = -1

    angle = 140

    return 0 < cavity < np.deg2rad(angle) or 0 < inverseCavity < np.deg2rad(angle)


def evaluateEllipse(ellipse, maxSideRatio, maxSize):
    sideDuple = ellipse[1]

    shortSide = np.min(sideDuple)
    longSide = np.max(sideDuple)

    if longSide / shortSide > maxSideRatio:
        return False

    if longSide > maxSize:
        return False

    return True


# main loop
for dataIndex in dataIndexes:
    start = time.time()
    segImOriginal = segData[dataIndex]

    cv2.imshow("original", segImOriginal)

    segIm = np.copy(segImOriginal)

    segIm[segIm < 0.5] = 0.0
    segIm[segIm > 0.0] = 1.0

    print(segIm.min())
    print(segIm.max())

    segIm *= 255.0/segIm.max()

    segIm = segIm.astype(np.uint8)

    #cv2.imshow("tresh", segIm)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(segIm, cv2.MORPH_OPEN, kernel, iterations=2)
    #opening = segIm

    #cv2.imshow("open", opening)

    D = ndimage.distance_transform_edt(opening)

    localMax = peak_local_max(D, indices=False, min_distance=0, labels=opening)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=opening)

    print(D.shape)
    print(D.min())
    print(D.max())

    D_disp = np.copy(D)

    D_disp *= 255.0/D_disp.max()

    D_disp = D_disp.astype(np.uint8)

    #cv2.imshow("D", D_disp)


    segImColor = np.copy(segImOriginal)

    segImColor *= 255

    segImColor = segImColor.astype(np.uint8)

    segImColor = np.dstack((segImColor, segImColor, segImColor))

    #testLabels = [143, 180, 236, 199, 97]
    #testLabels = [21]

    testLabels = np.unique(labels)

    breakFlag = False

    contourCount = 0
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(segImOriginal.shape, dtype="uint8")
        mask[labels == label] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        convertedCnt = np.asarray(convertContours(cnts[0]))
        approxCnt = skimage.measure.approximate_polygon(convertedCnt, tolerance=0.75)
        approxCnt = fuse(approxCnt, 5)

        cvContours = np.asarray([approxCnt])

        if label not in testLabels:
            continue

        cv2.drawContours(segImColor, cvContours, 0, (255, 0, 255), thickness=1)

        cavityApprox = concaveExtraction([], approxCnt)

        pointList = {}

        pointsTotal = len(approxCnt)
        pointsMiddle = int(np.round(pointsTotal / 2))

        for index, obj in enumerate(approxCnt):

            if len(approxCnt) < 3:
                continue

            dirContainer = cavityApprox[index]

            dir = dirContainer["angel"]

            prePt = tuple(dirContainer["pre"])
            curPt = tuple(dirContainer["cur"])
            nexPt = tuple(dirContainer["nex"])

            dirMin = -np.deg2rad(140)
            dirMax = np.deg2rad(140)

            # if dirMin > dir > dirMax:
            #    continue

            clr = (0, 255, 0)

            # if lineInContour(approxCnt, [pt1, pt2]) or dirMin > dir > dirMax:
            if not lineInContour(approxCnt, [prePt, nexPt]) and dirMin < dir < dirMax:
                clr = (0, 0, 255)
                cv2.circle(segImColor, curPt, 2, clr, thickness=-1)

                pointList[index] = {"point": curPt, "conc": dir}

                #opposite = (index + pointsMiddle) % pointsTotal

                #pointList[opposite] = approxCnt[opposite]

                #cv2.circle(segImColor, approxCnt[opposite], 3, (0, 255, 0), thickness=-1)

                #cv2.line(segImColor, prePt, nexPt, clr, thickness=1)

        if len(pointList.keys()) == 1:
            pointIndex = list(pointList.keys())[0]
            pointConcavity = pointList[pointIndex]["conc"]
            opposite = (pointIndex + pointsMiddle) % pointsTotal

            if evaluateCavityForMirror(pointConcavity):
                pointList[opposite] = approxCnt[opposite]

                clr = (0, 0, 255)
                cv2.circle(segImColor, approxCnt[opposite], 2, clr, thickness=-1)
            else:
                clr = (0, 255, 0)

            #cv2.putText(segImColor, "degs{}".format(int(np.rad2deg(pointConcavity))), approxCnt[opposite],
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #            clr, 1)

        slicedList = sliceList(approxCnt, pointList.keys())

        print("slicedList", slicedList)

        for l in slicedList:
            print("controur len", len(l))

            if len(l) < 2:
                continue

            cvContours = np.asarray([l])

            drawRectangle = False

            if len(l) < 5:
                # continue
                # print("Pre lin: ", l)
                # l = linearExpandLine(l, 5)
                # print("Post lin: ", l)
                # sys.exit(0)
                # breakFlag = True
                drawRectangle = True
            else:
                print("FFS", cvContours)
                el = cv2.fitEllipse(cvContours)
                drawRectangle = not evaluateEllipse(el, 4, 50)

                # cl = (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))

                                                                             

            #cl = (r.randint(32, 255), r.randint(32, 255), r.randint(32, 255))
            cl = (0, 255, 0)
            if drawRectangle:
                rect = cv2.minAreaRect(cvContours)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #cv2.drawContours(segImColor, cvContours, 0, cl, thickness=1)
            #else:
            #    cv2.ellipse(segImColor, el, cl, 1)

            cv2.line(segImColor, l[-1], l[0], (0, 0, 0), 2)

            contourCount += 1

            if breakFlag:
                break

        if breakFlag:
            break

    end = time.time()

    if logCSV:
        with open(csvFile, 'a', newline='') as file:
            ellapsed = end - start
            writer = csv.writer(file)
            writer.writerow([dataIndex, len(np.unique(labels)), contourCount, ellapsed])
        #((x, y), r) = cv2.minEnclosingCircle(cvContours)
        #cv2.putText(segImColor, "#{}".format(label), (int(x) + 10, int(y) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        #break

    cv2.imshow("original-colour", segImColor)

    cv2.imwrite(filePrefix + str(dataIndex) + ".png", segImColor)

    #cv2.imshow("cells", segIm)

cv2.waitKey()
cv2.destroyAllWindows()
