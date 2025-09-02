from process import convert

if __name__ == '__main__':
	convert('./sample-data/input.mp4', './sample-data/output.mp4', [f'./sample-data/{_}.jpg' for _ in range(6)])