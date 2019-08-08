import torch as t

class Criterion:
    def forward(inp, tar):
        #inp : batch_Size*outputlayer_nodes **double tensor
        #tar : batch_Size*1 corresponding labels
        bsize = inp.size()[0]
        inp = inp - t.max(inp, 1, True)[0]
        exinp = t.exp(inp)
        # print(tar)
        # print(exinp)
        deninp = t.sum(exinp, 1)
        numinp = exinp[t.arange(bsize), tar]
        # print(numinp)
        # print(deninp)
        soft_scores = numinp/deninp
        # print(soft_scores)
        loss = t.sum(t.log(soft_scores))
        return -loss/bsize


    def backward(inp, tar):
        bsize = inp.size()[0]
        gradIn = t.zeros(inp.size(), dtype = t.float64)
        gradIn[t.arange(bsize),tar] = -1
        
        inp = inp - t.max(inp, 1, True)[0]
        expinp = t.exp(inp)
        deninp = t.sum(expinp, 1, True)
        gradIn = gradIn + (expinp/deninp)
        return gradIn/bsize