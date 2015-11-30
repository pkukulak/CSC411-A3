from util import load_data
import matplotlib.pyplot as plt
from skimage import data
from skimage.viewer import ImageViewer


if __name__ == "__main__":
	data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
	image = data_in[0,:].reshape(32, 32)
	viewer = ImageViewer(image)
	viewer.show()