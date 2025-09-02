import numpy as np
import cv2

def convert(videoPathIn, videoPathOut, imgsIn):
    # load imgs
    imgs = []

    for imgPath in imgsIn:
        imgs.append(cv2.resize(cv2.imread(imgPath), (20, 20)))

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
                # Convert to int16 to avoid underflow
                frameArea = frame[y:y+imgHeight, x:x+imgWidth].astype(np.int16)
                deltas = np.array(imgs, dtype=np.int16)

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