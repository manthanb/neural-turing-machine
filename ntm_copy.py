import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class NTM(nn.Module):

	def __init__(self, n_in, n_h, n_out, N, M):

		super(NTM,self).__init__()

		self.n_in = n_in
		self.n_h = n_h
		self.n_out = n_out
		self.N = N
		self.M = M

		self.controller = nn.LSTMCell(n_in, n_h)

		self.v_fc = nn.Linear(n_h, n_out)
		self.zeta_fc = nn.Linear(n_h, 2*(M+5)+2*M)
		self.read_fc = nn.Linear(M, n_out)

		self.loss_function = nn.BCELoss()

	def _init_params(self):
		h_init = torch.FloatTensor(np.random.randn(1,self.n_h))
		c_init = torch.FloatTensor(np.random.randn(1,self.n_h))
		memory = torch.FloatTensor(np.zeros((self.N, self.M))+1e-6)
		read_weights = write_weights = F.sigmoid(torch.FloatTensor(np.random.randn(1,self.N)))
		return h_init, c_init, memory, read_weights, write_weights

	def _get_params(self, zeta):
		return zeta[0, 0:self.M+5].view(1,-1), zeta[0, self.M+5:].view(1,-1)

	def _cosine_similarity(self, k, mem):
		k = k.view(1, -1)
		return F.cosine_similarity(mem+1e-16, k+1e-16, dim=-1)

	def _convolve_shift(self, w_g, s):
		w_tild = torch.zeros(w_g.size())
		i = 0
		while(i < len(w_g[0])):
			w_tild[0][i] = w_g[0][int(i%len(w_g[0]))]*s[0] + w_g[0][int((i+1)%len(w_g[0]))]*s[1] + w_g[0][int((i+2)%len(w_g[0]))]*s[2]
			i += 1
		return w_tild

	def _sharpen(self, w_tild, gamma):
		w = w_tild ** gamma
		w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
		return w

	def _address(self, head_params, weights, memory):

		k = head_params[0,:self.M].view(1,-1)
		g = F.sigmoid(head_params[0, self.M])
		s = F.softmax(head_params[0, self.M+1:self.M+1+3])
		gamma = 1 + F.softplus(head_params[0,-1])

		similarity = self._cosine_similarity(k, memory)
		w_c = F.softmax(similarity)
		w_g = g * w_c + (1-g) * weights
		w_tild = self._convolve_shift(w_g, s)
		weights = self._sharpen(w_tild, gamma)

		return weights

	def _write_to_memory(self, write_params, write_weights, memory):

		erase_vector = write_params[0, self.M+5:self.M+5+self.M].view(1,-1)
		add_vector = write_params[0, self.M+5+self.M:].view(1,-1)

		mem_tild = memory * ( torch.tensor(np.zeros((self.N,self.M))+1).float() - torch.mm(torch.transpose(write_weights, 0, 1), erase_vector)) 
		memory = mem_tild + torch.mm(torch.transpose(write_weights, 0, 1), add_vector)

		return memory

	def _read_from_memory(self, read_weights, memory):
		return torch.mm(read_weights, memory)

	def forward(self, x, h_controller, c_controller, read_weights, write_weights, memory):
		
		h_controller, c_controller = self.controller(x,(h_controller,c_controller))
		v = self.v_fc(c_controller)
		zeta = self.zeta_fc(c_controller)
		
		read_params, write_params = self._get_params(zeta)
		read_weights, write_weights = self._address(read_params, read_weights, memory), self._address(write_params, write_weights, memory)

		memory = self._write_to_memory(write_params, write_weights, memory)
		read_weights = self._read_from_memory(read_weights, memory)

		y = v + self.read_fc(read_weights)

		out = F.log_softmax(y,dim=1)

		return h_controller, c_controller, memory, out

	def recurrent_forward(self, x):

		h_init, c_init, memory, read_weights, write_weights = self._init_params()

		h_controller = h_init.expand(x.size()[0],self.n_h)
		c_controller = c_init.expand(x.size()[0],self.n_h)
		zeros = torch.tensor(np.zeros((x.size()[0],self.n_in))).float()

		for i in range(x.size()[2]):
			h_controller, c_controller, memory, out = self.forward(x[:,:,i], h_controller, c_controller, read_weights, write_weights, memory)

		seq = []
		for i in range(x.size()[2]):
	 		h_controller, c_controller, memory, out = self.forward(zeros, h_controller, c_controller, read_weights, write_weights, memory)
	 		seq.append(out.unsqueeze(2))

		return torch.cat(seq,2)


def loadData(N):
    data = []
    for i in range(N):
        seq = np.zeros((4,32))
        for j in range(32):
            val = np.random.randint(4)
            seq[val,j] = 1
        
        data.append(seq)
    return torch.tensor(np.array(data)).float()

def main():

	train, test = loadData(1000), loadData(100)
	ntm = NTM(4, 32, 4, 10, 8)

	for epoch in range(1200):

		ntm.zero_grad()
		out = ntm.recurrent_forward(train)
		loss = -torch.mean(train*out)
		loss.backward()
		optimizer = torch.optim.Adam( ntm.parameters(), lr=(0.01/((epoch/500)+1)) )
		optimizer.step()

		print(epoch, loss)

	# torch.save(ntm.state_dict(), 'ntm_copy.pt')
	# ntm.load_state_dict(torch.load('ntm_copy.pt'))
	# ntm.eval()

	incorrect = 0
	seq = ntm.recurrent_forward(test).detach()
	test = test.numpy()
	predicted = np.zeros(np.shape(seq))

	for i in range(len(seq)):
		indices = np.argmax(seq[i], axis=0)
		for j in range(len(indices)):
			predicted[i, indices[j], j] = 1
			if not np.array_equal(predicted[i,:,j], test[i,:,j]): incorrect += 1

	# print("test:")
	# print(test)
	# print("predicted:")
	# print(predicted

	accuracy = ((32*100) - incorrect) / (32*100)
	print(incorrect, accuracy)

main()
