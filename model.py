import torch as t
import Criterion
import math
import os

class Model:
    def __init__(self, alpha, batchsize, epochs):
        self.alpha = alpha
        self.batchsize = batchsize
        self.epochs = epochs
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
    
    def train(self, trainX, trainY, validX = None, validY = None, saveModel=False, loadModel=False, modelName= None):

        if loadModel:
            model = t.load("./" + modelName + "/" + modelName)
            k,i = 0,0
            for l in self.layers:
                if type(l).__name__ == "RNN":
                    self.layers[i].Whh = model[k]
                    self.layers[i].Whx = model[k+1]
                    k+=2
                    i+=1
                elif type(l).__name__ == "Linear":
                    self.layers[i].W = model[k]
                    self.layers[i].B = model[k+1]
                    k+=2
                    i+=1
            print("Model loaded")
        # print(trainX.shape)
        for epoch in range(self.epochs):
            n = trainX.shape[0]
            ind = t.randperm(n)
            X = trainX[ind, :, :]
            Y = trainY[ind]
            trainl = 0
            vall = 0
            trainacc  = 0

            numB = int(math.ceil(float(X.shape[0]) / self.batchsize))
            for batchNum in range(numB):
                XBatch = X[batchNum*self.batchsize: (batchNum+1)*self.batchsize, :, :]
                YBatch = Y[batchNum*self.batchsize: (batchNum+1)*self.batchsize]

                print(XBatch)
                output = self.forward(XBatch)

                print(output)
                trainl+=Criterion.Criterion.forward(output, YBatch)
                # print(trainl)
                _, predlabels = t.max(output, 1)

                acc = t.eq(predlabels, YBatch).sum()
                trainacc+=acc

                gradOut = Criterion.Criterion.backward(output, YBatch)
                print("------------gradOut-----------")
                print(gradOut)
                self.backward(gradOut)

            trainacc /= (numB*batchNum)
            print("Epoch ", epoch, " Training Loss=", trainl, " Training Accuracy=", trainacc)

            if validX is not None and validY is not None:
                val_out = self.forward(validX)
                vall+= Criterion.Criterion.forward(val_out, validY)
                _, val_labels = t.max(val_out, 1)

                val_acc = t.eq(val_labels, validY).sum()
                val_acc /= len(validY)
                print("Validation set Accuracy: ", val_acc, "%")

            if saveModel:
                model = []
                for l in self.layers:
                    if type(l).__name__ == "RNN":
                        model.append(l.Whh)
                        model.append(l.Whx)
                    elif type(l).__name__ == "Linear":
                        model.append(l.W)
                        model.append(l.B)
                os.mkdir(modelName)
                t.save(model, "./" + modelName + "/" + modelName)
                print("Model saved")

    def forward(self, inp):
        for l in self.layers:
            if (type(l).__name__ == "Linear"):
                output = l.forward(inp[:, :, -1])
                self.linear_act = inp[:,:,-1]
            else:
                nex_inps = l.forward(inp)
                inp = nex_inps
        return output

    def backward(self, gradOut):
        seq_len = self.layers[-2].seq_len
        temp = self.layers[-1].backward(self.alpha, self.linear_act, gradOut)
        gradOut = t.zeros(temp.shape[0], temp.shape[1], seq_len + 1, dtype = t.float64)
        gradOut[:,:,-1] = temp
        print("-----------temp-------------")
        print(temp)
        for i in range(len(self.layers) - 2, -1, -1):
            gradOut = self.layers[i].backward(self.alpha, gradOut)

        
        
        
    