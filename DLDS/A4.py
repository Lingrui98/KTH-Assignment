import numpy as np
import string

DEBUG = 0
# d: number of unique characters, length of input vectors
# m: length of hidden state vectors
# C/K: length of output probability vectors, should be equal to d

eta = 0.1
gamma = 0.9
hid_m = 100
seq_length = 25

MAX_EPOCH = 1

def read_txt(name):
    with open(name, 'r') as file:
        text = file.read()
    return text

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
        print (len(data))
        self.in_size = in_size
        self.out_size = out_size
        self.hid_size = hid_size
        self.eta = eta
        self.seq_length = seq_length

        self.fieldnames = ['W','U','V','b','c']

        self.h0 = np.reshape(np.zeros(hid_size), (hid_size,1))
        self.hprev = self.h0.copy()

        self.para = {'W': np.random.normal(0, 1, (hid_size, hid_size)) * 0.1,
                     'U': np.random.normal(0, 1, (hid_size,  in_size)) * 0.1,
                     'V': np.random.normal(0, 1, (out_size, hid_size)) * 0.1,
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

        self.CHECK_GRADS = True

    def softmax(self,s):
        s = np.exp(s)
        a = np.sum(s)
        s = s / a
        return s

    def string_to_matrix(self, str):
        l = len(str) # should be 25
        for i in range(l):
            v = ind_to_vec(char_to_ind[str[i]])
            if i == 0:
                M = v
            else:
                M = np.hstack([M,v])
        assert M.shape==(v.shape[0],l)
        return M


    def generate_texts(self, length, h0, x0):
        n = length
        texts = []
        h = h0
        x = np.reshape(x0, (self.in_size,1))
        ind = to_ind(x)
        texts.append(self.ind_to_char[ind])
        for i in range(n):
            a = np.dot(self.para['W'],h) + np.dot(self.para['U'],x) + self.para['b']
            h = np.tanh(a)
            o = np.dot(self.para['V'],h) + self.para['c']
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

    def grad_clip(self):
        for f in self.fieldnames:
            shape = self.grad[f].shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.grads[f][i][j] = max(min(self.grads[f][i][j], 5), -5)
        return

    def get_data(self, e):
        X = self.data[e   : e+self.seq_length]
        Y = self.data[e+1 : e+self.seq_length+1]
        X = self.string_to_matrix(X)
        Y = self.string_to_matrix(Y)
        return X, Y

    def forward_pass(self, para, hprev, X, Y):
        A = []
        H = [hprev]
        P = []
        L = 0.0
        for i in range(seq_length):
            x = np.reshape(X[:,i], (self.in_size , 1))
            y = np.reshape(Y[:,i], (self.out_size, 1))
            a = np.dot(para['W'],hprev) + np.dot(para['U'],x) + para['b']
            h = np.tanh(a)
            o = np.dot(para['V'],h) + para['c']
            p = self.softmax(o)
            A.append(a)
            H.append(h)
            P.append(p)
            l = -np.log(np.dot(y.T,p))[0][0]
            L += l
            self.hprev = h
        return L, H, A, P

    def compute_gradients(self, X, Y, H, A, P):
        grads = {'W': np.reshape(np.zeros(self.hid_size*self.hid_size), (self.hid_size,self.hid_size)),
                 'U': np.reshape(np.zeros(self.in_size *self.hid_size), (self.hid_size,self.in_size)),
                 'V': np.reshape(np.zeros(self.out_size*self.hid_size), (self.out_size,self.hid_size)),
                 'b': np.reshape(np.zeros(self.hid_size), (self.hid_size,1)),
                 'c': np.reshape(np.zeros(self.out_size), (self.out_size,1))
        }

        G = []
        for i in range(self.seq_length):
            y = np.reshape(Y[:,i], (self.out_size, 1))
            p = P[i]
            g = -(y-p).T
            G.append(g)
        for i in range(self.seq_length):
            grads['V'] += np.dot(np.reshape(G[i], (self.out_size,1)), np.reshape(H[i+1], (1, self.hid_size)))

        dht = np.dot(np.reshape(G[self.seq_length-1], (1,self.out_size)), self.para['V'])
        # print ((1-np.square(np.tanh(A[self.seq_length-1]))).shape)
        diagat = np.diag(np.reshape((1-np.square(np.tanh(A[self.seq_length-1]))), (self.hid_size,)))
        # print (diagat.shape)
        dat = np.dot(dht,diagat)
        grads['W'] += np.dot(dat.T, H[self.seq_length-1].T)
        grads['U'] += np.dot(dat.T, np.reshape(X[:,self.seq_length-1], (1, self.in_size)))
        for i in range(self.seq_length-1):
            t = self.seq_length - 2 -i
            dht = np.dot(G[t],self.para['V']) + np.dot(dat, self.para['W'])
            diagat = np.diag(np.reshape((1-np.square(np.tanh(A[t]))), (self.hid_size,)))
            dat = np.dot(dht,diagat)
            grads['W'] += np.dot(dat.T, H[t].T)
            grads['U'] += np.dot(dat.T, np.reshape(X[:,t], (1, self.in_size)))

        return grads

    def ComputeGradsNum(self, X, Y, h):
        grads = {'W': np.reshape(np.zeros(self.hid_size*self.hid_size), (self.hid_size,self.hid_size)),
                 'U': np.reshape(np.zeros(self.in_size *self.hid_size), (self.hid_size,self.in_size)),
                 'V': np.reshape(np.zeros(self.out_size*self.hid_size), (self.out_size,self.hid_size)),
                 'b': np.reshape(np.zeros(self.hid_size), (self.hid_size,1)),
                 'c': np.reshape(np.zeros(self.out_size), (self.out_size,1))
        }
        hprev = self.hprev
        for f in self.fieldnames:
            print ('Computing numeriacal gradient for %s' % (f))
            for r in range(self.para[f].shape[0]):
                for c in range(self.para[f].shape[1]):
                    para_try = self.para.copy()
                    para_try[f][r][c] = self.para[f][r][c] - h
                    l1, _, __, ___ = self.forward_pass(para_try, hprev, X, Y)
                    para_try[f][r][c] = self.para[f][r][c] + h
                    l2, _, __, ___ = self.forward_pass(para_try, hprev, X, Y)
                    grads[f][r][c] = (l2-l1) / (2*h)
        return grads

    def backward_pass(self, X, Y, H, A, P):
        hprev = self.hprev
        grads = self.compute_gradients(X, Y, H, A, P)
        h_prev_after = self.hprev
        if self.CHECK_GRADS:
            self.hprev = hprev
            num_grads = self.ComputeGradsNum(X,Y,1e-4)
            self.hprev = h_prev_after
            D = {}
            S = {}
            RD = {}
            for f in self.fieldnames:
                D[f] = grads[f] - num_grads[f]
                S[f] = np.abs(grads[f]) + np.abs(num_grads[f])
                RD[f] = np.mean(np.abs(D[f])) / np.mean(S[f])
                print ('Relative difference in %s is ' % (f), RD[f])
        # update
        for f in self.fieldnames:
            self.grad[f] -= self.eta * grads[f]
        return

    def train_network(self):
        smooth_loss = 0
        for i in range(MAX_EPOCH):
            e = 0
            loss = 0
            ite = 0
            while(e+self.seq_length <= len(self.data)):
                X, Y = self.get_data(e)
                L, H, A, P = self.forward_pass(self.para, self.hprev, X, Y)
                loss = L
                # print ('loss in epoch %d' % i)
                self.backward_pass(X, Y, H, A, P)

                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                if (e % 10000 == 0):
                    print ('e=', e)
                    print ('Smooth loss in iter %d is %.2f' % (i+1, smooth_loss))
                    x0 = ind_to_vec(int(np.round(np.random.random()*total_num_char)))
                    self.generate_texts(200, self.h0, x0)

                e += seq_length
                ite += 1
            self.h_prev = self.h0


        print ('Training done for %d iterations\n' % (MAX_EPOCH))
        x0 = ind_to_vec(int(np.round(np.random.random()*total_num_char)))
        self.generate_texts(200, self.h0, x0)

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
    ind0 = int(np.round(np.random.random()*total_num_char))
    x0 = ind_to_vec(ind0)
    # print ('Initial input character is %s\n' % (ind_to_char[ind0]))
    # texts = R.generate_texts(200,h0,x0)
    # print ('After initialization, generated text from random x0 is:\n\n%s' % (texts))
    R.train_network()
