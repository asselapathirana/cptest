# USAGE
# python predict.py --image images/dog.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# python predict.py --image concrete/Negative/00014.jpg --model output/concrete.model --label-bin output/concrete_lb.pickle --width 64 --height 64

# import the necessary packages

import random
from numpy import array


def load_model_and_labels(model_name,label_bin):
	"""fake results for GUI testing"""
	class dumb():
		pass
	graph = dumb()
	model = dumb()
	lb = dumb()
	lb.classes_=["Negative","Positive"]
	return graph, model, lb




def predict2(graph, model, image, x_, y_):
	"""fake results for GUI testing"""
	p=random.random()
	preds=array([[p,1-p]])
	return preds

