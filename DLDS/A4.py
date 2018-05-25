import numpy as np
import string
import copy
import matplotlib.pyplot as plt

DEBUG = 0
# d: number of unique characters, length of input vectors
# m: length of hidden state vectors
# C/K: length of output probability vectors, should be equal to d

eta = 0.1
gamma = 0.9
hid_m = 100
seq_length = 25

MAX_EPOCH = 10
CHECK = False

def plot_loss(l):
    ite = []
    for i in range(len(l)):
        ite.append(i+1)
    fig = plt.plot(ite,l,'b')
    plt.title('The training process')
    plt.xlabel('Iterations')
    plt.ylabel('Smooth loss')
    plt.legend(loc='best')
    plt.savefig('loss.png')
    plt.show()

def read_txt(name):
    with open(name, 'r') as file:
        texts = file.read()
    return texts

def get_unique_chars(str):
    char_list = list(set(str))
    char_list.sort()
    return char_list

# Returns a python dictionary from characters to numbers
def set_map(list_char):
    char_to_ind = {}
    ind_to_char = {}
    for i in range(len(list_char)):
        char_to_ind[list_char[i]] = i
        ind_to_char[i] = list_char[i]
    return char_to_ind, ind_to_char


def ind_to_vec(a, total_num_char=80):
    vec = np.zeros(total_num_char)
    # assert type(a) == int
    vec[a] = 1
    vec = np.reshape(vec, (total_num_char,1))
    return vec

def char_to_vec(a, char_to_ind, total_num_char=80):
    assert type(a) == str and len(a) == 1
    vec = np.zeros(total_num_char)
    vec[char_to_ind[a]] = 1
    vec = np.reshape(vec, (total_num_char,1))
    return vec


def to_ind(vec):
    a = np.where(vec==1)[0]
    assert a.shape[0]==1
    return a[0]

class RNN(object):
    def __init__(self, data, in_size, out_size, hid_size, eta, seq_length, dict):
        super(RNN, self).__init__()
        self.data = data
        # print (len(data))
        self.in_size = in_size
        self.out_size = out_size
        self.hid_size = hid_size
        self.eta = eta
        self.seq_length = seq_length

        self.fieldnames = ['W','U','V','b','c']

        self.h0 = np.reshape(np.zeros(hid_size), (hid_size,1))
        self.hprev = self.h0.copy()

        self.para = {'W': np.random.normal(0, 1, (hid_size, hid_size)) * 0.01,
                     'U': np.random.normal(0, 1, (hid_size,  in_size)) * 0.01,
                     'V': np.random.normal(0, 1, (out_size, hid_size)) * 0.01,
                     'b': np.reshape(np.zeros(hid_size), (hid_size,1)),
                     'c': np.reshape(np.zeros(out_size), (out_size,1))}
        self.grad = {'W': np.reshape(np.zeros(self.hid_size*self.hid_size), (self.hid_size,self.hid_size)),
                     'U': np.reshape(np.zeros(self.in_size *self.hid_size), (self.hid_size,self.in_size)),
                     'V': np.reshape(np.zeros(self.out_size*self.hid_size), (self.out_size,self.hid_size)),
                     'b': np.reshape(np.zeros(hid_size), (hid_size,1)),
                     'c': np.reshape(np.zeros(out_size), (out_size,1))}

        # for name in self.fieldnames:
        #     print (self.para[name])

        self.char_to_ind = dict[0]
        self.ind_to_char = dict[1]

        self.CHECK_GRADS = CHECK

        self.ADA = True
        self.CLIP = False

    def init_para(self):
        para  = {'W': np.reshape(np.zeros(self.hid_size*self.hid_size), (self.hid_size,self.hid_size)),
                 'U': np.reshape(np.zeros(self.in_size *self.hid_size), (self.hid_size,self.in_size)),
                 'V': np.reshape(np.zeros(self.out_size*self.hid_size), (self.out_size,self.hid_size)),
                 'b': np.reshape(np.zeros(self.hid_size), (self.hid_size,1)),
                 'c': np.reshape(np.zeros(self.out_size), (self.out_size,1))
        }
        return para

    def softmax(self,s):
        s = np.exp(s)
        a = np.sum(s)
        s = s / a
        return s

    def string_to_matrix(self, str):
        # print (str)
        l = len(str) # should be 25
        for i in range(l):
            v = ind_to_vec(char_to_ind[str[i]])
            if i == 0:
                M = v
            else:
                M = np.hstack([M,v])
        assert M.shape==(v.shape[0],l)
        return M


    def generate_texts(self, length, h0, x0, para):
        n = length
        texts = []
        h = h0
        x = np.reshape(x0, (self.in_size,1))
        ind = to_ind(x)
        texts.append(self.ind_to_char[ind])
        for i in range(n):
            a = np.dot(para['W'],h) + np.dot(para['U'],x) + para['b']
            h = np.tanh(a)
            o = np.dot(para['V'],h) + para['c']
            p = self.softmax(o)

            s = np.cumsum(p)
            rand = np.random.random()
            ind = np.where(s>rand)[0][0]
            x = ind_to_vec(ind)

            texts.append(self.ind_to_char[ind])
        texts = ''.join(texts)
        print (texts)
        return
        # return texts

    def grad_clip(self, grads):
        clipped = self.init_para()
        for f in self.fieldnames:
            shape = self.grad[f].shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    clipped[f][i][j] = max(min(grads[f][i][j], 5), -5)
        return clipped

    def get_data(self, e):
        X = self.data[e   : e+self.seq_length]
        Y = self.data[e+1 : e+self.seq_length+1]
        X = self.string_to_matrix(X)
        Y = self.string_to_matrix(Y)
        return X, Y

    def forward_pass(self, para, hp, X, Y):
        # Using input parameter hprev instead of self.prev
        # Does not change self.hprev
        assert X.shape[1] == self.seq_length
        assert Y.shape[1] == self.seq_length

        h_in = hp
        H = [hp]
        A = []
        P = []
        L = 0.0
        for i in range(self.seq_length):
            x = np.reshape(X[:,i], (self.in_size , 1))
            y = np.reshape(Y[:,i], (self.out_size, 1))
            a = np.dot(para['W'],H[i]) + np.dot(para['U'],x) + para['b']
            h = np.tanh(a)
            o = np.dot(para['V'],h) + para['c']
            p = self.softmax(o)
            # print ('p shape', p.shape)
            # print ('y_sum', np.sum(y))
            H.append(h)
            A.append(a)
            P.append(p)
            # print ('is', np.dot(y.T,p))
            l = -np.log(np.dot(y.T,p))[0][0]
            L += l
            # print ('l=', l)
        # self.hprev = h_in
        # print ('L=', L)
        # print ('len H', len(H))
        return L, H, A, P

    def compute_gradients(self, X, Y, H, A, P):
        # Does not change self.hprev
        grads = self.init_para()

        G = []
        for i in range(self.seq_length):
            y = np.reshape(Y[:,i], (self.out_size, 1))
            p = P[i]
            g = -(y-p).T
            # print ('g=',g)
            # print ('sum g =', np.sum(g))
            G.append(g)
        for i in range(self.seq_length):
            grads['V'] += np.dot(np.reshape(G[i], (self.out_size,1)), np.reshape(H[i+1], (1, self.hid_size)))
            grads['c'] += np.reshape(G[i], (self.out_size,1))
        assert len(G) == self.seq_length
        dht = np.dot(np.reshape(G[-1], (1,self.out_size)), self.para['V'])
        # print ((1-np.square(np.tanh(A[self.seq_length-1]))).shape)
        diagat = np.diag(np.reshape((1-np.tanh(A[-1])*np.tanh(A[-1])), (self.hid_size,)))
        # print (diagat.shape)
        dat = np.dot(dht,diagat)
        grads['b'] += np.reshape(dat, (self.hid_size,1))
        grads['W'] += np.dot(dat.T, H[-2].T)
        grads['U'] += np.dot(dat.T, np.reshape(X[:,-1], (1, self.in_size)))
        for t in reversed(range(self.seq_length-1)):
            # print (t)
            dht = np.dot(G[t],self.para['V']) + np.dot(dat, self.para['W'])
            diagat = np.diag(np.reshape((1-np.tanh(A[t])*np.tanh(A[t])), (self.hid_size,)))
            dat = np.dot(dht,diagat)
            grads['b'] += np.reshape(dat, (self.hid_size,1))
            grads['W'] += np.dot(dat.T, H[t].T)
            grads['U'] += np.dot(dat.T, np.reshape(X[:,t], (1, self.in_size)))
        # print (grads)
        return grads

    def ComputeGradsNum(self, X, Y, hp, h):
        # Calls the function forward_pass twice
        grads = self.init_para()
        # hprev = self.hprev
        # para = self.para.copy()
        for f in self.fieldnames:
            print ('Computing numeriacal gradient for %s' % (f))
            for r in range(self.para[f].shape[0]):
                for c in range(self.para[f].shape[1]):
                    para_try = copy.deepcopy(self.para)
                    para_try[f][r][c] = self.para[f][r][c] - h
                    # print ('para_try-[%s][%d][%d] = %f, para = %f' %(f,r,c,para_try[f][r][c],self.para[f][r][c]))
                    l1, _, __, ___ = self.forward_pass(para_try, hp, X, Y)
                    para_try = copy.deepcopy(self.para)
                    para_try[f][r][c] = self.para[f][r][c] + h
                    # print ('para_try+[%s][%d][%d] = %f, para = %f' %(f,r,c,para_try[f][r][c],self.para[f][r][c]))
                    l2, _, __, ___ = self.forward_pass(para_try, hp, X, Y)
                    grads[f][r][c] = (l2-l1) / (2*h)
        # print (grads['b'],grads['c'])
        # for f in self.fieldnames:
        #     assert (para[f] == self.para[f]).all()
        return grads

    def backward_pass(self, X, Y, H, A, P):
        grads = self.compute_gradients(X, Y, H, A, P)
        if self.CHECK_GRADS:
            # self.hprev = hprev
            hprev = self.hprev
            num_grads = self.ComputeGradsNum(X,Y,hprev,1e-4)
            D = {}
            S = {}
            RD = {}
            for f in self.fieldnames:
                D[f] = grads[f] - num_grads[f]
                if f == 'W':
                    print (D[f][0],'\n', grads[f][0], '\n', num_grads[f][0])
                S[f] = np.abs(grads[f]) + np.abs(num_grads[f])
                RD[f] = np.mean(np.abs(D[f])) / np.mean(S[f])
                print ('Relative difference in %s is' % (f), RD[f])
        # update
        # for f in self.fieldnames:
        #     self.para[f] -= self.eta * grads[f]
        return grads

    def train_network(self):
        ite = 0
        list_loss = []
        smooth_loss = 0

        min_L = 10000000
        best_para = self.init_para()

        for i in range(MAX_EPOCH):
            e = 0
            m = self.init_para()
            while(e+self.seq_length <= len(self.data)-1):
                X, Y = self.get_data(e)
                L, H, A, P = self.forward_pass(self.para, self.hprev, X, Y)
                # print ('loss in epoch %d' % i)
                grads = self.backward_pass(X, Y, H, A, P)
                if self.CLIP:
                    grads = self.grad_clip(grads)
                # AdaGrad
                if self.ADA:
                    for f in self.fieldnames:
                        m[f] += np.square(grads[f])
                        self.para[f] -= (self.eta / np.sqrt(m[f] + 1e-3)) * grads[f]
                else:
                    for f in self.fieldnames:
                        self.para[f] -= self.eta * grads[f]

                smooth_loss = 0.999 * smooth_loss + 0.001 * L
                list_loss.append(smooth_loss)

                if (L < min_L):
                    best_para = copy.deepcopy(self.para)
                    min_L = L


                if ((ite+1) % 10000 == 0):
                    print ('\nSmooth loss in epoch %d, iter %d is %.2f\n' % (i+1, ite+1, smooth_loss))
                    x0 = ind_to_vec(int(np.round(np.random.random()*total_num_char-1)))
                    self.generate_texts(200, self.h0, x0, self.para)

                e += self.seq_length
                # print (e)
                ite += 1
                self.hprev = H[-1]
            self.hprev = self.h0


        print ('\nTraining done for %d iterations\n' % (MAX_EPOCH))
        x0 = ind_to_vec(int(np.round(np.random.random()*(total_num_char-1))))
        print ('Minimum loss is %f, generating texts of length 1000 with the relative parameters...\n' % min_L)
        self.generate_texts(1000, self.h0, x0, best_para)
        return list_loss

if __name__ == '__main__':
    # Read the data into a consecutive sequence
    data = ''.join(read_txt('goblet_book.txt'))
    # Get the list of unique characters
    characters = get_unique_chars(data)
    # Get the dictionaries between characters and indices
    char_to_ind, ind_to_char = set_map(characters)
    total_num_char = len(char_to_ind)

    #print (len(characters))

    if DEBUG == 1:
        for i in range(len(characters)):
            assert(char_to_ind[ind_to_char[i]] == i)
        print ('Checked!')

    if DEBUG == 2:
        vec = char_to_vec('\t',char_to_ind)
        vec2 = ind_to_vec(0)
        print (vec, '\n', vec2)
        # vec_false1 = to_vec('fsad',char_to_num, total_num_char)
        # vec_false2 = to_vec(1.1,char_to_num, total_num_char)
        # print vec_false1, vec_false2

    if DEBUG == 3:
        vec = char_to_vec('a',char_to_ind)
        ind = to_ind(vec)
        assert ind_to_char[ind] == 'a'
        print ('Checked!')

    R = RNN(data, total_num_char, total_num_char, hid_m, eta, seq_length, [char_to_ind,ind_to_char])
    h0 = np.random.normal(0,1,(hid_m,1))
    ind0 = int(np.round(np.random.random()*total_num_char-1))
    x0 = ind_to_vec(ind0)
    # print ('Initial input character is %s\n' % (ind_to_char[ind0]))
    # texts = R.generate_texts(200,h0,x0)
    # print ('After initialization, generated text from random x0 is:\n\n%s' % (texts))
    l = R.train_network()
    plot_loss(l)
