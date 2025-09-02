from multiprocessing import Pool, shared_memory
import numpy as np
import cv2
import os

def imgLoad(path, size=None):
    imgs = []

    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            imgs.append(cv2.imread(path+file))

    if size:
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], size)

    return imgs

def processRow(args):
    yStart, yEnd, frameShape, frameName, imgsName, numImgs, imgHeight, imgWidth = args

    sharedFrameMem = shared_memory.SharedMemory(name=frameName)
    sharedImgsMem = shared_memory.SharedMemory(name=imgsName)

    frame = np.ndarray(frameShape, dtype=np.uint8, buffer=sharedFrameMem.buf)
    imgs = np.ndarray((numImgs, imgHeight, imgWidth, frameShape[2]), dtype=np.int16, buffer=sharedImgsMem.buf)

    videoWidth = frameShape[1]

    for xStart in range(0, videoWidth, imgWidth):
        xEnd = min(xStart+imgWidth, videoWidth)
        frameArea = frame[yStart:yEnd, xStart:xEnd].astype(np.int32)

        deltas = np.empty((numImgs, (yEnd-yStart), (xEnd-xStart), frameShape[2]), dtype=np.int32)
        for d in range(len(deltas)):
            np.subtract(imgs[d, :(yEnd-yStart), :(xEnd-xStart)], frameArea, out=deltas[d])
            np.square(deltas[d], out=deltas[d])
            #np.abs(deltas[d], out=deltas[d])

        minIdx = np.argmin(np.sum(deltas, axis=(1,2,3)))
        frame[yStart:yEnd, xStart:xEnd] = imgs[minIdx, :(yEnd-yStart), :(xEnd-xStart)].astype(np.uint8)

    sharedFrameMem.close()
    sharedImgsMem.close()

def convertMulti(videoPathIn, videoPathOut, imgs):
    if len(set(img.shape for img in imgs)) != 1:
        raise ValueError('Images need to have same shape (dimension / colour)')

    imgHeight, imgWidth, channels = imgs[0].shape

    # load videoIn
    videoIn = cv2.VideoCapture(videoPathIn)

    fps = int(videoIn.get(cv2.CAP_PROP_FPS))
    videoWidth  = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totalFrames = int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not videoIn.isOpened():
        raise ValueError(f'Cannot open video {videoPathIn}')

    # make videOut
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoOut = cv2.VideoWriter(videoPathOut, fourcc, fps, (videoWidth, videoHeight))

    # memory
    imgsArr = np.array(imgs, dtype=np.int16)
    sharedImgsMem = shared_memory.SharedMemory(create=True, size=imgsArr.nbytes)
    sharedImgs = np.ndarray(imgsArr.shape, dtype=np.int16, buffer=sharedImgsMem.buf)
    sharedImgs[:] = imgsArr[:]

    # multicore
    pool = Pool()

    counter = 1 
    while 1:
        ret, frame = videoIn.read()

        if not ret:
            break

        sharedFrameMem = shared_memory.SharedMemory(create=True, size=frame.nbytes)
        sharedFrame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=sharedFrameMem.buf)
        sharedFrame[:] = frame[:]

        tasks = []
        for yStart in range(0, videoHeight, imgHeight):
            yEnd = min(yStart + imgHeight, videoHeight)
            tasks.append((yStart, yEnd, frame.shape, sharedFrameMem.name, sharedImgsMem.name, len(imgs), imgHeight, imgWidth))

        pool.map(processRow, tasks)

        videoOut.write(sharedFrame)

        # memory
        sharedFrameMem.close()
        sharedFrameMem.unlink()

        print(f'{counter}/{totalFrames}')
        counter += 1

    # memory
    pool.close()
    pool.join()
    sharedImgsMem.close()
    sharedImgsMem.unlink()

    # video writer
    videoIn.release()
    videoOut.release()

def convertSingle(videoPathIn, videoPathOut, imgs):
    if len(set(img.shape for img in imgs)) != 1:
        raise ValueError('Images need to have same shape (dimension / colour)')

    imgHeight, imgWidth, channels = imgs[0].shape

    # load videoIn
    videoIn = cv2.VideoCapture(videoPathIn)
    
    if not videoIn.isOpened():
        raise ValueError(f'Cannot open video {videoPathIn}')

    # create videoOut
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = int(videoIn.get(cv2.CAP_PROP_FPS))
    videoWidth  = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoOut = cv2.VideoWriter(videoPathOut, fourcc, fps, (videoWidth, videoHeight))

    # frame iterator and constructor
    counter = 1
    while 1:
        ret, frame = videoIn.read()

        if not ret:
            break

        frameOut = np.zeros((videoHeight, videoWidth, channels), dtype=np.uint8)

        for y in range(0, videoHeight-imgHeight, imgHeight):
            for x in range(0, videoWidth-imgWidth, imgWidth):
                # Convert to int32 to avoid underflow
                frameArea = frame[y:y+imgHeight, x:x+imgWidth].astype(np.int32)
                deltas = np.array(imgs, dtype=np.int32)

                for d in range(len(deltas)):
                    np.subtract(deltas[d], frameArea, out=deltas[d])
                    #np.square(deltas[d], out=deltas[d])
                    np.abs(deltas[d], out=deltas[d])

                frameOut[y:y+imgHeight, x:x+imgWidth] = imgs[np.argmin(np.array([np.sum(delta) for delta in deltas]))]

        videoOut.write(frameOut)

        print(f'Frame {counter}/{int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))}')
        counter += 1

    videoIn.release()
    videoOut.release()