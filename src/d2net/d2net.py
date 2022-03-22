#!/usr/bin/env python
import logging
import sys
import time
import select

import socket
from threading import Thread
from torch.multiprocessing import Process, Queue, Event, Pipe
from queue import Empty

import numpy as np

# import imageio
# from PIL import Image
import cv2
import pyquaternion
import torch
import math

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale as processTorch

trtavailable = False
try:
    from lib.pyramidNew import process_multiscale as processTRT
    from torch2trt import TRTModule

    trtavailable = True
except ImportError:
    pass
    #logging.warning("Failed to import TRT. Only torch is supported!")

# Hardcoded for now
MODELPATH = "/models/"
preprocessing = "caffe"
TCP_PORT = 8076


# Initialization
def initializeTorch(processNum, outfile):
    # Hardcode for now
    model_file = MODELPATH + "d2_tf.pth"
    use_relu = True

    # CUDA
    use_cuda = torch.cuda.is_available()

    if processNum < torch.cuda.device_count():
        devname = "cuda:" + str(processNum)
        device = torch.device(devname)
        model = D2Net(
            model_file=model_file,
            use_relu=use_relu,
            use_cuda=use_cuda,
            device=device
        )
        return device, model


def initializeTRT(processNum, fp16, outfile):
    # Hardcode for now
    use_relu = True
    if (fp16 == 'on'):
        model_file = MODELPATH + "d2_trt_denseFeatureExtraction_fp16ON_AWS.pth"
    else:
        model_file = MODELPATH + "d2_trt_denseFeatureExtraction_fp16OFF_AWS.pth"

    # CUDA
    use_cuda = torch.cuda.is_available()

    if processNum < torch.cuda.device_count():
        devname = "cuda:" + str(processNum)
        device = torch.device(devname)

        model = TRTModule()
        model.load_state_dict(torch.load(model_file))
        model.to(device)
        return device, model


def processImage(image, localDevice, localModel, method):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    input_image = preprocess_image(
        image,
        preprocessing=preprocessing
    )
    start = time.time()
    with torch.no_grad():
        if (method == 'trt'):
            keypoints, scores, descriptors = processTRT(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=localDevice
                ),
                localModel,
                scales=[1]
            )
        else:
            keypoints, scores, descriptors = processTorch(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=localDevice
                ),
                localModel,
                scales=[1]
            )
    elapsed = (time.time() - start)

    return keypoints, descriptors, scores


def toKeypoints(kps):
    result = []
    for kp in kps:
        result.append(cv2.KeyPoint(x=kp[0], y=kp[1], _size=1))
    return result


def pickTopDescriptorsAndTransform(keypoints, descriptors, scores, params, transform, bounds):
    # Input image coordinates
    scores = scores.reshape(len(keypoints), 1)

    # set keypoints to be pairs of x,y
    keypoints = keypoints[:, [1, 0]]

    if transform is not None:
        # apply transform, rotation first
        rotation = transform[0:2, 0:2]
        keypoints = np.matmul(rotation, keypoints.transpose())

        # then the translation
        translation = np.array([[transform[0, 2], transform[1, 2]]] * keypoints.shape[1])
        keypoints = keypoints.transpose() + translation

    # stack keypoints, descriptors and scores
    data = np.hstack((keypoints, descriptors, scores))

    # filter keypoints that are outside the image!
    indices = (keypoints[:, 0] >= 0) * (keypoints[:, 0] < bounds[0]) * (keypoints[:, 1] >= 0) * (keypoints[:, 1] < bounds[1])
    data = data[indices]

    # sort by score
    data = data[data[:, -1].argsort()]

    # choose top features
    if len(keypoints) > params.maxNrFeatures:
        data = data[-params.maxNrFeatures:]

        # get keypoints
    keypoints = data[:, :2]

    # get descriptors
    if params.integerFeature:
        descriptors = np.clip(data[:, 2:-1], 0.0, 0.5) * 510
        descriptors = descriptors.astype(np.uint8)
    else:
        descriptors = data[:, 2:-1]
        descriptors = descriptors.astype(np.float32)

    return keypoints, descriptors


def pickTopDescriptors(keypoints, descriptors, scores, maxNrFeatures):
    # Input image coordinates
    keypoints = keypoints[:, [0, 1]]
    scores = scores.reshape(len(keypoints), 1)

    keypoints = np.hstack((keypoints, scores))
    descriptors = np.hstack((descriptors, scores))

    keypoints = keypoints[keypoints[:, -1].argsort()]
    descriptors = descriptors[descriptors[:, -1].argsort()]

    if (len(keypoints) > maxNrFeatures):
        keypoints = keypoints[-maxNrFeatures:, :-1]
        descriptors = descriptors[-maxNrFeatures:, :-1]
    else:
        keypoints = keypoints[:, :-1]
        descriptors = descriptors[:, :-1]

    # Encode the full return data to be processed by the client
    data = np.array([float(len(keypoints)), float(descriptors.shape[1])])
    data = np.append(data.flatten(), np.append(keypoints.flatten(), descriptors.flatten()).flatten())
    return data


def processRawImage(rawImage, maxNrFeatures, device, model, method, outfile):
    image = cv2.imdecode(np.fromstring(rawImage, dtype=np.uint8), 1)
    keypoints, descriptors, scores = processImage(image, device, model, method)
    return pickTopDescriptors(keypoints, descriptors, scores, maxNrFeatures)


##############################################################################

def prepareImage(image, params):
    height, width = image.shape[:2]
    scaledBounds = (width, height)
    scaling = 1.0

    if params.targetFocalLength and params.intrinsics:  # can't scale based on focal length if the images focal length is unknown
        avgFocalLength = (params.intrinsics.fx + params.intrinsics.fy) / 2.0
        scaling = params.targetFocalLength / avgFocalLength
    elif params.maxSideLength:  # disregard maxSideLength
        scaleH = float(params.maxSideLength) / float(height)
        scaleW = float(params.maxSideLength) / float(width)
        scaling = min(scaleH, scaleW)

    # TODO implement scaling based on focal lengths
    # if a scaling factor has been determined
    if scaling < 1.0:  # do not upscale images
        height = int(math.ceil(height * scaling))
        width = int(math.ceil(width * scaling))
        scaledBounds = (width, height)

        image = cv2.resize(image, scaledBounds, interpolation=cv2.INTER_AREA)
        # cv2.imwrite("/data/test.jpg", image)

        # adapt intrinsics
        if params.intrinsics:
            params.intrinsics.cx *= scaling
            params.intrinsics.cy *= scaling
            params.intrinsics.fx *= scaling
            params.intrinsics.fy *= scaling

    rotMat = None
    # if we need to rotate
    if params.rotation:  # rotation in opencv...
        imgcenter = (width / 2.0, height / 2.0)
        rotMat = cv2.getRotationMatrix2D(imgcenter, params.rotation, 1.0)

        # need to manually compute the new width and height without cropping the rotated image
        cos = np.abs(rotMat[0][0])
        sin = np.abs(rotMat[0][1])

        # TODO test this, I think it's wrong
        newWidth = int(math.ceil((height * cos) + (width * sin)))
        newHeight = int(math.ceil((height * sin) + (width * cos)))

        rotMat[0][2] += (newHeight / 2) - imgcenter[0]
        rotMat[1][2] += (newWidth / 2) - imgcenter[1]

        image = cv2.warpAffine(image, rotMat, (newHeight, newWidth))
        # cv2.imwrite("/data/testrot.jpg", image)

        rotMat = np.array([[rotMat[0, 0], rotMat[0, 1], rotMat[0, 2]],
                              [rotMat[1, 0], rotMat[1, 1], rotMat[1, 2]],
                              [0, 0, 1]])
        rotMat = np.linalg.inv(rotMat)

    return image, scaledBounds, rotMat


class D2Parameters():
    def __init__(self, rawImage, maxNrFeatures=2000, rotation=None, intrinsics=None, maxSideLength=None, targetFocalLength=None,
                 integerFeature=False):
        self.rawImage = rawImage
        self.maxNrFeatures = maxNrFeatures
        self.rotation = rotation
        self.intrinsics = intrinsics
        self.maxSideLength = maxSideLength
        self.targetFocalLength = targetFocalLength
        self.integerFeature = integerFeature


def processProcedure(q, exit, i, method, fp16, outpath, ready):
    # this runs in its own process
    if outpath is None:
        outfile = sys.stdout
    else:
        outfile = open(outpath + "_" + str(i) + ".log", "a")

    if (method == 'trt'):
        device, model = initializeTRT(i, fp16, outfile)
    else:
        device, model = initializeTorch(i, outfile)

    ready.set()
    while not exit.is_set():
        try:
            [params, output] = q.get(block=True, timeout=1)  # wait for a second at most (so we can be interrupted)
        except Empty:
            continue

        try:
            image = cv2.imdecode(np.fromstring(params.rawImage, dtype=np.uint8), 1)

            # scale and rotate image
            image, bounds, rotMat = prepareImage(image, params)
            keypoints, descriptors, scores = processImage(image, device, model, method)

            keypoints, descriptors = pickTopDescriptorsAndTransform(keypoints, descriptors, scores, params, rotMat, bounds)
            output.send([keypoints, descriptors, params.intrinsics])

        except:
            logging.exception('Error processing image')
            output.send([None, None, None])


class D2Procedure():
    def __init__(self, method="torch", fp16="off", outfile=None):
        torch.multiprocessing.set_start_method('spawn', force=True)
        self.outfile = outfile

        if method == "trt" and trtavailable == False:
            method = "torch"
            print("ERROR: trt was specified, but could not be imported! Make sure tensorRT is correctly installed! Falling back to torch",
                  flush=True, file=self.outfile)

        self.method = method
        self.processes = []
        self.queue = Queue()
        self.exit = Event()  # for stopping the child-processes
        self.ready = []

        # first set up all processes
        for i in range(torch.cuda.device_count()):
            readyEvent = Event()
            p = Process(target=processProcedure, args=(self.queue, self.exit, i, self.method, fp16, outfile, readyEvent))
            p.start()
            self.processes.append(p)
            self.ready.append(readyEvent)

        ready = False
        while not ready:
            ready = True
            for re in self.ready:
                if not re.is_set():
                    ready = False
                    break

    def stop(self):
        self.exit.set()

    def execute(self, params):
        rec, output = Pipe()  # for receiving the feature descriptors
        self.queue.put([params, output])
        return rec.recv()


##############################################################################
# Multithreaded Python server : TCP Server Socket Thread Pool
class Client():

    def __init__(self, ip, port, conn, dev, model, method, outfile):
        print(ip, port, conn, file=outfile)
        self.ip = ip
        self.port = port
        self.conn = conn
        self.device = dev
        self.model = model
        self.method = method
        self.outfile = outfile
        print("[+] New server socket thread started for " + ip + ":" + str(port), file=self.outfile)

    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def run(self):
        print(self.device, "RUN() START", flush=True, file=self.outfile)
        if (self.conn.fileno() == -1): return
        try:
            dataSize = self.recvall(20)
            # print("D2: rec'd ", len(dataSize), " bytes of image size data", flush=True)
            nbytes = int(dataSize)
        except:
            print("D2: Error in d2netServer.py, when receiving image size", flush=True, file=self.outfile)
            return

        # handshake
        s1 = self.conn.send(np.array([1]))
        # print("D2: send ", s1, " handshake 1 data", flush=True)

        try:
            data = self.recvall(nbytes)
            # print("D2: rec'd ", len(data), " bytes of image data", flush=True)
        except:
            print("D2: Error in d2netServer.py, when receiving image data", flush=True, file=self.outfile)
            return
        if not data: return

        # handshake
        s2 = self.conn.send(np.array([1]))
        # print("D2: send ", s2, " handshake 2 data", flush=True)

        try:
            maxNum = self.recvall(20)
            # print("D2: rec'd ", len(maxNum), " bytes of param data", flush=True)
        except:
            print("D2: Error in d2netServer.py, when receiving maximum nr of points", flush=True, file=self.outfile)
            return
        if not maxNum: return

        featureData = processRawImage(data, int(maxNum), self.device, self.model, self.method, self.outfile)

        if (self.conn.fileno() == -1): return
        try:
            data = featureData.astype('<f').tostring()
            self.conn.send(data)
        except:
            print("D2: Error in d2netServer.py, when sending keypoints", flush=True, file=self.outfile)
            return
        self.conn.close()
        print(self.device, "RUN() STOP", flush=True, file=self.outfile)


def process(q, exit, i, method, fp16, outpath):
    # this runs in its own process

    if outpath is None:
        outfile = sys.stdout
    else:
        outfile = open(outpath + "_" + str(i) + ".log", "a")

    if (method == 'trt'):
        device, model = initializeTRT(i, fp16, outfile)
    else:
        device, model = initializeTorch(i, outfile)

    while not exit.is_set():
        try:
            (conn, (ip, port)) = q.get(block=True, timeout=1)  # wait for a second at most (so we can be interrupted)
        except Empty:
            continue

        newClient = Client(ip, port, conn, device, model, method, outfile)
        newClient.run()


class MainD2Thread(Thread):

    def __init__(self, port, method="torch", fp16="off", outfile=None):
        Thread.__init__(self)
        torch.multiprocessing.set_start_method('spawn', force=True)

        if method == "trt" and trtavailable == False:
            method = "torch"
            print("ERROR: trt was specified, but could not be imported! Make sure tensorRT is correctly installed! Falling back to torch",
                  flush=True, file=self.outfile)

        self.method = method
        self.processes = []
        self.queue = Queue()
        self.exit = Event()  # for stopping this thread and the child-processes

        # first set up all processes
        for i in range(torch.cuda.device_count()):
            p = Process(target=process, args=(self.queue, self.exit, i, self.method, fp16, outfile))
            p.start()
            self.processes.append(p)

        # Multithreaded Python server : TCP Server Socket Program Stub
        TCP_IP = '0.0.0.0'

        self.tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcpServer.bind((TCP_IP, port))

    def stop(self):
        self.exit.set()

    def run(self):
        while not self.exit.is_set():
            self.tcpServer.listen(4)

            r, w, e = select.select([self.tcpServer], [], [], 1)  # wait for 1 second until the socket is available
            for s in r:
                # (conn, (ip, port)) = s.accept()
                self.queue.put(s.accept())

        for p in self.processes:
            p.join()
