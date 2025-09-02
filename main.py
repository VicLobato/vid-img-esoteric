from process import convertSingle, convertMulti, imgLoad

if __name__ == '__main__':
	imgs = imgLoad('./sample-data/', (50, 50))
	convertMulti('./sample-data/input.mp4', './sample-data/output.mp4', imgs)