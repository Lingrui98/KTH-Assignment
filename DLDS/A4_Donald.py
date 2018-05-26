import numpy as np
import string
import copy
import matplotlib.pyplot as plt
import json

DEBUG = 0
# d: number of unique characters, length of input vectors
# m: length of hidden state vectors
# C/K: length of output probability vectors, should be equal to d

eta = 0.1
gamma = 0.9
hid_m = 200
seq_length = 141

MAX_EPOCH = 3
MAX_GRAD = 3000
CHECK = False
ADA = True
CLIP = False


def plot_loss(l):
    ite = []
    for i in range(len(l)):
        ite.append(i+1)
    fig = plt.plot(ite,l,'b')
    plt.title('The training process')
    plt.xlabel('Iterations')
    plt.ylabel('Smooth loss')
    # plt.legend(loc='best')
    plt.savefig('Donald_loss.png')
    plt.show()

def read_tweets():
    # Read tweets of length at most 140
    tweets = []
    max = 140
    count = 0
    for i in range(10):
        j = 2009+i
        json_str = open('condensed_%s.json' % (str(j))).read()
        tweet = json.loads(json_str)
        for k in range(len(tweet)):
            n = len(tweet[k]['text'])
            s = tweet[k]['text']
            if n > max:
                continue
            if n < 141:
                for l in range(n,141):
                    s += '\a'
            # print (len(s))
            count += 1
            tweets.append(s)
    print ('Totally %d tweets of Donald Trump read' % count)
    return tweets


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


def ind_to_vec(a, total_num_char=398):
    vec = np.zeros(total_num_char)
    vec[a] = 1
    vec = np.reshape(vec, (total_num_char,1))
    return vec

def char_to_vec(a, char_to_ind, total_num_char=398):
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

        self.char_to_ind = dict[0]
        self.ind_to_char = dict[1]

        self.CHECK_GRADS = CHECK

        self.ADA = ADA
        self.CLIP = CLIP

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
        l = len(str) # should be 25
        for i in range(l):
            v = ind_to_vec(char_to_ind[str[i]])
            if i == 0:
                M = v
            else:
                M = np.hstack([M,v])
        assert M.shape==(v.shape[0],l)
        return M


    def generate_tweets(self, length, h0, x0, para):
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

        index = texts.find('\a')
        if index != -1:
            texts = texts[index:]

        print (texts)
        return

    def grad_clip(self, grads):
        clipped = self.init_para()
        for f in self.fieldnames:
            shape = self.grad[f].shape
            if np.sum(np.square(grads[f])) > MAX_GRAD:
                clipped[f] = MAX_GRAD * grads[f] / np.sum(np.square(grads[f]))
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
            H.append(h)
            A.append(a)
            P.append(p)
            l = -np.log(np.dot(y.T,p))[0][0]
            L += l
        return L, H, A, P

    def compute_gradients(self, X, Y, H, A, P):
        # Does not change self.hprev
        grads = self.init_para()

        G = []
        for i in range(self.seq_length):
            y = np.reshape(Y[:,i], (self.out_size, 1))
            p = P[i]
            g = -(y-p).T
            G.append(g)
        for i in range(self.seq_length):
            grads['V'] += np.dot(np.reshape(G[i], (self.out_size,1)), np.reshape(H[i+1], (1, self.hid_size)))
            grads['c'] += np.reshape(G[i], (self.out_size,1))
        assert len(G) == self.seq_length

        dht = np.dot(np.reshape(G[-1], (1,self.out_size)), self.para['V'])
        diagat = np.diag(np.reshape((1-np.tanh(A[-1])*np.tanh(A[-1])), (self.hid_size,)))
        dat = np.dot(dht,diagat)
        # The gradient of b, W and U contributed by the last character
        grads['b'] += np.reshape(dat, (self.hid_size,1))
        grads['W'] += np.dot(dat.T, H[-2].T)
        grads['U'] += np.dot(dat.T, np.reshape(X[:,-1], (1, self.in_size)))
        # Gradients contributed by all other characters
        for t in reversed(range(self.seq_length-1)):
            dht = np.dot(G[t],self.para['V']) + np.dot(dat, self.para['W'])
            diagat = np.diag(np.reshape((1-np.tanh(A[t])*np.tanh(A[t])), (self.hid_size,)))
            dat = np.dot(dht,diagat)
            grads['b'] += np.reshape(dat, (self.hid_size,1))
            grads['W'] += np.dot(dat.T, H[t].T)
            grads['U'] += np.dot(dat.T, np.reshape(X[:,t], (1, self.in_size)))
        return grads

    def ComputeGradsNum(self, X, Y, hp, h):
        # Calls the function forward_pass twice
        grads = self.init_para()
        for f in self.fieldnames:
            print ('Computing numeriacal gradient for %s' % (f))
            for r in range(self.para[f].shape[0]):
                for c in range(self.para[f].shape[1]):
                    para_try = copy.deepcopy(self.para)
                    para_try[f][r][c] = self.para[f][r][c] - h
                    l1, _, __, ___ = self.forward_pass(para_try, hp, X, Y)
                    para_try = copy.deepcopy(self.para)
                    para_try[f][r][c] = self.para[f][r][c] + h
                    l2, _, __, ___ = self.forward_pass(para_try, hp, X, Y)
                    grads[f][r][c] = (l2-l1) / (2*h)

        return grads

    def backward_pass(self, X, Y, H, A, P):
        grads = self.compute_gradients(X, Y, H, A, P)
        if self.CHECK_GRADS:
            hprev = self.hprev
            num_grads = self.ComputeGradsNum(X,Y,hprev,1e-4)
            D = {}
            S = {}
            RD = {}
            for f in self.fieldnames:
                D[f] = grads[f] - num_grads[f]
                if f == 'V':
                    print ('grads_V=\n',grads[f])
                S[f] = np.abs(grads[f]) + np.abs(num_grads[f])
                RD[f] = np.mean(np.abs(D[f])) / np.mean(S[f])
                print ('Relative difference in %s is' % (f), RD[f])
        return grads

    def train_network(self):
        ite = 0
        list_loss = []
        smooth_loss = 0

        min_L = 10000000
        best_para = self.init_para()

        m = self.init_para()

        for i in range(MAX_EPOCH):
            e = 0
            while(e+self.seq_length <= len(self.data)-1):
                X, Y = self.get_data(e)
                L, H, A, P = self.forward_pass(self.para, self.hprev, X, Y)
                grads = self.backward_pass(X, Y, H, A, P)

                # for f in self.fieldnames:
                #     print ('Grad of %s is %f' %(f,np.sum(np.square(grads[f]))))

                if self.CLIP:
                    grads = self.grad_clip(grads)
                # AdaGrad
                if self.ADA:
                    for f in self.fieldnames:
                        m[f] += np.square(grads[f])
                        self.para[f] -= (self.eta / np.sqrt(m[f] + 1e-8)) * grads[f]
                else:
                    for f in self.fieldnames:
                        self.para[f] -= self.eta * grads[f]

                smooth_loss = 0.999 * smooth_loss + 0.001 * L
                list_loss.append(smooth_loss)

                if (L < min_L):
                    best_para = copy.deepcopy(self.para)
                    min_L = L

                if ((ite+1) % 100 == 0):
                    print ('\nSmooth loss in epoch %d, iter %d is %.2f\n' % (i+1, ite+1, smooth_loss))
                    x0 = ind_to_vec(int(np.round(np.random.random()*total_num_char-1)))
                    self.generate_tweets(140, self.h0, x0, self.para)

                e += self.seq_length
                ite += 1
                self.hprev = self.h0
            self.hprev = self.h0


        print ('\nTraining done for %d iterations\n' % (MAX_EPOCH))
        x0 = ind_to_vec(int(np.round(np.random.random()*(total_num_char-1))))
        print ('Minimum loss is %f, generating texts of length 1000 with the relative parameters...\n' % min_L)
        self.generate_tweets(140, self.h0, x0, best_para)
        return list_loss

if __name__ == '__main__':
    # Read the data into a consecutive sequence
    all_tweets = ''.join(read_tweets())
    # Get the list of unique characters
    characters = get_unique_chars(all_tweets)
    print (characters, len(characters), ''.join(characters).find("\a"))
    # Get the dictionaries between characters and indices
    char_to_ind, ind_to_char = set_map(characters)
    total_num_char = len(char_to_ind)

    # # Initializing RNN network
    R = RNN(all_tweets, total_num_char, total_num_char, hid_m, eta, seq_length, [char_to_ind,ind_to_char])
    # read_tweets()
    l = R.train_network()
    plot_loss(l)
