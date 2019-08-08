import torch as t
import math
import torch as t
import torch.nn as nn

class RNN:
    def __init__(self, inp_dim, hid_dim, max_seq_len):
        self.seq_len = max_seq_len
        self.Whh = t.randn(hid_dim, hid_dim, dtype = t.float64)/ math.sqrt(hid_dim) 
        self.Whx = t.randn(hid_dim, inp_dim, dtype = t.float64)/ math.sqrt(inp_dim) # h(t) = Whh*h(t-1) + Whx*x(t-1) 

        self.hid_dim = hid_dim
        self.inp_dim = inp_dim
        self.gradWhh = t.zeros(hid_dim, hid_dim, dtype = t.float64)
        self.gradWhx = t.zeros(hid_dim, inp_dim, dtype = t.float64)
        
    def set_batchsize(self, size):
        self.batch_size = size

    def forward(self, inp_seq):
        batch_size = inp_seq.shape[0]
        self.output = t.zeros(batch_size, self.hid_dim, self.seq_len + 1, dtype = t.float64) 
        self.hid_states = t.zeros(batch_size, self.hid_dim, self.seq_len + 1, dtype = t.float64)
        self.hid_states[:,:,0] = t.zeros(batch_size, self.hid_dim, dtype = t.float64)
        self.grad_hid = t.zeros(batch_size, self.hid_dim, self.seq_len + 1, dtype = t.float64)
        self.grad_inp = t.zeros(batch_size, self.inp_dim, self.seq_len, dtype = t.float64)
        self.input = t.zeros(batch_size, self.inp_dim, self.seq_len, dtype = t.float64)

        self.input = inp_seq
        for i in range(self.seq_len):
            # print(self.input[:,:,i].shape)
            # print(self.Whx.shape)
            temp = t.matmul(inp_seq[:, :, i], t.transpose(self.Whx, 0, 1))
            self.output[:, :, i+1] = t.matmul(self.hid_states[:, :, i], t.transpose(self.Whh, 0, 1)) + temp
            m = nn.Tanh()
            self.hid_states[:, :, i+1] = m(self.output[:, :, i+1])
        return self.hid_states

    def backward(self, lr, list_grad_final_stat):
        #list_grad_final_state : [batchsiae, hid_dim, seq_len +1] gradients from above layer
        self.grad_hid[:, :, self.seq_len] = list_grad_final_stat[:, :, -1]
        for i in range(self.seq_len-1,-1, -1):
            m = nn.Tanh()
            temp = self.output[:, :, i] - t.max(self.output[:, :, i+1])
            grad_out = self.grad_hid[:, :, i+1] * (1 - (m(self.output[:, :,i+1])**2))
            # print("--------gradOut--------------")
            # print(grad_out)
            self.grad_hid[:, :, i] = t.matmul(grad_out, self.Whh) + list_grad_final_stat[:, :, i]
            grwhh = t.matmul(t.transpose(grad_out, 0, 1), self.hid_states[:, :, i])
            self.gradWhh += grwhh
            grwhx = t.matmul(t.transpose(grad_out, 0, 1), self.input[:, :, i])
            self.gradWhx += grwhx
            self.grad_inp[:, :, i] = t.matmul(grad_out, self.Whx)
        self.Whh = self.Whh - lr*self.gradWhh
        self.Whx = self.Whx - lr*self.gradWhx
        print("--------Weights------------")
        print(self.Whh)
        return self.grad_inp

class Linear:
    def __init__(self,in_nodes, out_nodes):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.output = None

        self.gradW = None
        self.gradB = None
        init_weights = t.randn(in_nodes + 1, out_nodes, dtype = t.float64)/math.sqrt(in_nodes + 1)
        self.W = init_weights[:-1, :]
        self.B = init_weights[-1, :]

    def forward(self, input):
        temp = t.matmul(input, self.W)
        self.output = temp + self.B
        return self.output

    def backward(self, lr, input, gradOut):
        gradinp = t.matmul(gradOut, t.transpose(self.W, 0, 1))
        self.gradW = t.matmul(t.transpose(input, 0 ,1), gradOut)
        self.gradB = t.sum(gradOut, dim=0)
        self.W -= lr*self.W
        self.B -= lr*self.B
        return gradinp

    # class Linear:
    # 	def __init__(self, in_nodes, out_nodes):
    # 		self.in_nodes = in_nodes
    # 		self.out_nodes = out_nodes

    # 		self.output = None
    # 		self.gradW = None
    # 		self.gradB = None
    #         init_weights = t.randn(in_nodes + 1, out_nodes)/math.sqrt(in_nodes + 1)
    #         self.W = init_weights[:-1,:]
    #         self.B = init_weights[-1, :]

    # 	def forward(self,input):
    # 		temp =  torch.matmul(input,self.W)
    # 		self.output = temp + self.B
    # 		return self.output

    # 	def backward(self, lr, input, gradOut):
    # 		# input :- batch_size * in_nodes
    # 		# gradOut :- batch_size * out_nodes(dE/dA)
    # 		# gradInp :- batch_size * in_nodes
    # 		gradInp = torch.matmul(gradOut,torch.transpose(self.W, 0, 1))
    # 		self.gradW = torch.matmul(torch.transpose(input, 0, 1),gradOut)
    # 		self.gradB = torch.sum(gradOut, dim=0)
    # 		self.W = self.W - lr*self.gradW
    # 		self.B = self.B - lr*self.gradB
    # 		return gradInp