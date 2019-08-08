import torch as t
import torchfile
import argparse
import model
import data
from RNN import *

num2idx = {}
idx2num = []

func = lambda x : int(num2idx[x])
funcl = lambda li : list(map(func, li))
def help_func(li_data):
	c = 0
	for i in range(len(li_data)):
		li_data[i] = li_data[i].split()
		# print(list_data[i])
		for j in li_data[i]:
			if j not in num2idx:
				# print(j)
				idx2num.append(j)
				num2idx[j] = len(idx2num) - 1
		if(len(li_data[i]) > c): c = len(li_data[i])
		# li_data[i] = list(map(int, li_data[i]))
		# li_data[i].append(len(li_data[i]))
	return li_data, c

def one_hot_encode(X, num_labels):
	tx = t.tensor(X)
	out = t.zeros(num_labels, len(X))
	return out.scatter_(0, tx.view(1, tx.shape[0]), 1)

def prep_sequence(train_data, valid_data, test_data):
	# train_data, valid_data -> lists
	train_out = [0]*(len(train_data))
	valid_out = [0]*(len(valid_data))

	tr_data, tr_max = help_func(train_data)
	va_data, va_max = help_func(valid_data)
	te_data, te_max = help_func(test_data)

	# print(train_data)
	train_data = list(map(funcl, tr_data))
	valid_data = list(map(funcl, va_data))
	test_data = list(map(funcl, te_data))
	# print(len(list))
	# print(c)
	# print(train_data)
	# print(num2idx)
	c = max(tr_max, va_max, te_max)
	train_out = t.zeros(len(train_data), 153, c)
	valid_out = t.zeros(len(valid_data), 153, c)
	test_out = t.zeros(len(test_data), 153, c)
	for i in range(len(train_data)):
		# print(i)
		temp = one_hot_encode(train_data[i], 153)
		# print(temp.shape)
		# train_out[i,:,:] = [0]*(c-len(train_data[i]) + 1) + train_data[i]
		train_out[i, :, :] = t.cat((t.zeros(153, c - len(train_data[i])), temp), 1)
	for i in range(len(valid_data)):
		temp = one_hot_encode(valid_data[i], 153)
		# valid_out[i] = [0]*(c-len(valid_data[i]) + 1) + valid_data[i]
		valid_out[i, :, :] = t.cat((t.zeros(153, c - len(valid_data[i])), temp), 1)
	for i in range(len(test_data)):
		temp = one_hot_encode(test_data[i], 153)
		# valid_out[i] = [0]*(c-len(valid_data[i]) + 1) + valid_data[i]
		if(c!=len(test_data[i])):
			test_out[i, :, :] = t.cat((t.zeros(153, c - len(test_data[i])), temp), 1)
		else:
			test_out[i, :, :] = temp
	return c, train_out, valid_out, test_out

parser = argparse.ArgumentParser()
# parser.add_argument("-modelName", help = "Name of the Model")
parser.add_argument("-tr_data", help = "training data")
parser.add_argument("-tr_labels", help = "test labels")
parser.add_argument("-va_data", help = "validation_data")
parser.add_argument("-va_labels", help = "validation_targets")
parser.add_argument("-te_data", help = "test_data")
parser.add_argument("-Modelname", help = "Name on the saved model")
args = parser.parse_args()

train_data = open(args.tr_data, "r")
valid_data = open(args.va_data, "r")
test_data = open(args.te_data, "r")
# data = torchfile.load(args.data)
# data = torch.tensor(data, dtype = torch.float)
# data = data.view(data.size()[0],data.size()[1] * data.size()[2])
seq_len, train_seq, valid_seq, test_seq = prep_sequence(train_data.read().splitlines(), valid_data.read().splitlines(), test_data.read().splitlines())
# print(idx2num)
# print(len(num2idx))
train_seq = t.tensor(train_seq, dtype=t.float64)
valid_seq = t.tensor(valid_seq, dtype=t.float64)
test_seq = t.tensor(test_seq, dtype=t.float64)
# print(seq_len, train_seq, valid_seq, test_seq)

train_labels = open(args.tr_labels, "r")
valid_labels = open(args.va_labels, "r")
train_labels = train_labels.read().splitlines()
valid_labels = valid_labels.read().splitlines()
train_labels = [int(i) for i in train_labels]
valid_labels = [int(i) for i in valid_labels]
train_labels = t.tensor(train_labels, dtype = t.long)
valid_labels = t.tensor(valid_labels, dtype = t.long)
train_labels = train_labels.view(train_labels.size()[0],1)
valid_labels = valid_labels.view(valid_labels.size()[0],1)
# print(labels)

nn = model.Model(0.1, 2, 30)
nn.addLayer(RNN(153, 64, 2948))
nn.addLayer(Linear(64, 2))
# nn.layers[0].set_weights(wei1, b1)

# nn.addLayer(Relu(6))

# nn.addLayer(Linear(10,3))
# nn.layers[2].set_weights(wei2,b2)
nn.train(train_seq, train_labels, valid_seq, valid_labels,
True, False, modelName=args.Modelname)

# nn.dispGradParam()

