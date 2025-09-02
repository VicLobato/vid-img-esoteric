from process import convertSingle, convertMulti, imgLoad

if __name__ == '__main__':
	imgs = imgLoad('./data/road-signs/', (50, 50))
	convertMulti('./data/car.mp4', './data/car_out.mp4', imgs)