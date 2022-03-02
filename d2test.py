#!/usr/bin/env python

import argparse

import d2net
from pyquaternion import Quaternion
import numpy as np
import math
import cv2
import json


# Argparse
parser = argparse.ArgumentParser(description = 'D2-net server script.')
parser.add_argument('--port', type = int, default = 8076, required=False, help='The port this server should listen on')
parser.add_argument('--method', default = 'torch', choices=['torch', 'trt', 'torchMulti'], required=False, help='The model type we would like to use')
parser.add_argument('--fp16', default = 'off', choices=['on', 'off'], required=False, help='Floating point 16 optimization')
parser.add_argument('--img', required=True)
parser.add_argument('--pose', required=True)
args = parser.parse_args()

def readJSONFile(filename):
    result = None
    with open(filename) as file:
        result = json.load(file)
    return result

def getGravityRotation(q):
    y = np.array(q.rotate([0,1,0]))
    z = np.array(q.rotate([0,0,1]))

    dist = z[1]

    gravity = np.array([0,1,0]) - dist * np.array(z)

    norm = np.linalg.norm(gravity)
    if norm < 0.0001:
        return 0.0
    gravity /= norm

    cosAngle = y.dot(gravity)
    cosAngle = max(min(cosAngle, 1.0), -1.0)
    actualAngle = math.degrees( math.acos(cosAngle) )

    if z.dot( np.cross(y, gravity) ) < 0:
        return -actualAngle;
    return actualAngle;

def main():
    method = args.method
    fp16 = args.fp16
    port = args.port

    impath   = args.img
    jsonpath = args.pose

    metainf = readJSONFile(jsonpath)

    d2Extractor = d2net.D2Procedure(method, fp16)


    r = metainf['pose']['orientation']
    q = Quaternion(r['w'], r['x'], r['y'], r['z'])
    a = getGravityRotation(q)
    intrinsics = metainf['intrinsics']

    # image data
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # should convert to single channel but doesn't for this opencv version
    ret, rawImage = cv2.imencode('.jpg', img)
    print(ret)

    params = d2net.D2Parameters(rawImage, maxSideLength=640, intrinsics=intrinsics, rotation=a)
    featureData, intr = d2Extractor.execute(params)#rawImage, a, intrinsics, 2000, 640)
    print(featureData.shape)
    d2Extractor.stop()

if __name__ == '__main__':
    main()

