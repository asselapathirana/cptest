# USAGE
# python predict.py --image images/dog.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# python predict.py --image concrete/Negative/00014.jpg --model output/concrete.model --label-bin output/concrete_lb.pickle --width 64 --height 64

# import the necessary packages
import tensorflow as tf
from keras.models import load_model
#from tensorflow import get_default_graph
from pyimagesearch.smallvggnet import dumb
import pickle
import cv2
from urllib.request import urlopen
import numpy as np


def load_model_and_labels(model_name, label_bin):
	# load the model and label binarizer
	print("[INFOx] loading network and label binarizer...")
	with open(label_bin, "rb") as f_:
		lb = pickle.loads(f_.read())
	model = load_model(model_name)
	graph = tf.Graph()
	

	return graph, model, lb

def predict(graph, model, lb, image,x_,y_):
	output = image.copy()
	preds = predict2(image, x_, y_, graph, model)
	# find the class label index with the largest corresponding
	# probability
	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	# draw the class label + probability on the output image
	text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
	cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(0, 0, 255), 2)

	return output

def predict2(graph, model, image, x_, y_):
	
	image = cv2.resize(image, (x_,y_))
	# scale the pixel values to [0, 1]
	image = image.astype("float") / 255.0
	# working with a CNN -- don't flatten the
	# image, simply add the batch dimension
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))
	# make a prediction on the image
	#with graph.as_default():
	preds = model.predict(image)
	return preds


if __name__ == "__main__":
	graph, model, lb = load_model_and_labels('./data/concrete_best.model','./data/concrete_lb.pickle')
	# load the input image and resize it to the target spatial dimensions
	image = cv2.imread('800px-Yin_yang.svg.png')
	output = predict(graph, model, lb, image, 64, 64)	
	# show the output image
	cv2.imshow("Image", output)
	cv2.waitKey(0)	