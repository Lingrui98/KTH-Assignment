import cPickle
import numpy as np
import os
import copy
import matplotlib.pyplot as plt


batch_size = 100
eta = 0.1
lamda = 0.005
epoch = 60

# initialization = 'He'
initialization = 'Xavier'
# initialization = 0

n_layers = 3

out_size = 10
in_size = 3072
m_nodes = [in_size,256,128,out_size]
rho = 0.99

CHECK = False

SVM = 0
CrossEntropy = 1

def max(a,b):
    return a if a > b else b

def plot(a,b,c):
    ep = []
    for i in xrange(len(a)):
        ep.append(i+1)

    l1 = plt.plot(ep,a,'r',label='train')
    l2 = plt.plot(ep,b,'g',label='validate')
    l3 = plt.plot(ep,c,'b',label='test')
    plt.title('The training process')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.legend(loc='best')
    plt.show()

class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, n_layers, m_nodes, init, rho, eta=0.1, lamda=0, epoch=100, batch_size=batch_size, reader=None):
        super(Classifier, self).__init__()
        self.out_size = m_nodes[-1]
        self.in_size = m_nodes[0]
        self.n_layers = n_layers
        self.m_nodes = m_nodes # number of hidden nodes
        self.rho = rho
        self.learning_rate = eta
        self.lambda_= lamda
        self.max_epoch = epoch

        W = []
        for i in xrange(n_layers):
            sigma = 0.0
            if init == 'Xavier':
                sigma = 1 / np.sqrt(float(m_nodes[i]))
            else:
                if init == 'He':
                    sigma = np.sqrt(2 / float(m_nodes[i]))
                else:
                    sigma = 0.001
            print "sigma = %f in layer %d" % (sigma, i+1)
            W.append(np.random.normal(0,sigma, (m_nodes[i+1],m_nodes[i])))

        self.W = W

        b = []
        for i in xrange(n_layers):
            b.append(np.reshape(np.array([0.0 for j in xrange(m_nodes[i+1])]), (m_nodes[i+1],1)))

        self.b = b

        self.batch_size = batch_size
        # self.RW = np.sum(np.square(self.W)) * self.lambda_
        self.reader = reader

    def softmax(self,s):
        s = np.exp(s)
        a = np.sum(s)
        s = s / a
        return s

# Forward pass
    def EvaluateClassifier(self, X, W, b):
        n = X.shape[1]
        P = np.zeros((self.out_size,n))
        P = np.reshape(P,(self.out_size,n))

        for i in xrange(n):
            s = [np.reshape(X[:,i],(self.in_size,1))]
            h = [s[0]]
            for j in xrange(self.n_layers):
                s.append(np.dot(W[j],h[j]) + b[j])
                h.append(np.abs((np.abs(s[j+1])+s[j+1]) / 2)) # max(0,s)
            # print s[self.n_layers]
            p = self.softmax(s[self.n_layers])
            if i == 0:
                P = p
            else:
                P = np.hstack([P,p])
        # print P.shape
        return P

    def ComputeCost(self, X, Y, W, b, lambda_):
        n = X.shape[1]
        P = self.EvaluateClassifier(X, W, b)
        L = 0.0
        for i in xrange(n):
            L += -np.log(np.dot(np.reshape(Y[:,i],(1,self.out_size)),np.reshape(P[:,i],(self.out_size,1))))[0][0]
        J = L / float(n)
        R = 0.0
        for i in xrange(self. n_layers):
            R += np.sum(np.square(W[i]))
        J += lambda_ * R
        return J

    def ComputeAccuracy(self, X, Y, W, b):
        n = X.shape[1]
        P = self.EvaluateClassifier(X, W, b)
        labels = np.array([np.argmax(Y[:,i]) for i in xrange(n)])
        out = np.array([np.argmax(P[:,i]) for i in xrange(n)])
        num = 0.0
        for i in xrange(n):
            if labels[i]==out[i]:
                num += 1.0
        acc = num / float(n)
        return acc

    def ComputeGradients(self, X, Y, P, W, b, lambda_):
        n = X.shape[1]
        grad_W = []
        grad_b = []
        for i in xrange(self.n_layers):
            grad_W.append(np.reshape(np.zeros((self.m_nodes[i+1],self.m_nodes[i])), (self.m_nodes[i+1],self.m_nodes[i])))
            grad_b.append(np.reshape(np.zeros(self.m_nodes[i+1]), (self.m_nodes[i+1],1)))



        for i in xrange(n):
            s = [np.reshape(X[:,i],(self.in_size,1))]
            h = [s[0]]
            diag = [[]]
            for j in xrange(self.n_layers):
                s.append(np.dot(W[j],h[j]) + b[j])
                h.append(np.abs((np.abs(s[j+1])+s[j+1]) / 2)) # max(0,s1)

                if j < self.n_layers-1:
                    diag.append(np.reshape(np.zeros((self.m_nodes[j+1],self.m_nodes[j+1])), (self.m_nodes[j+1],self.m_nodes[j+1])))
                # print diag
                    for k in xrange(self.m_nodes[j+1]):
                        if s[j+1][k] > 0:
                            (diag[j+1])[k][k] = 1

            # print diag
            g = -(np.reshape(Y[:,i],(self.out_size,1)) - np.reshape(P[:,i],(self.out_size,1)))

            for j in xrange(self.n_layers):
                k = self.n_layers - j
                # print k
                # print b[k-1].shape
                grad_b[k-1] += g
                grad_W[k-1] += np.dot(g,np.reshape(h[k-1],(1,self.m_nodes[k-1])))

                if k > 1:
                    # print g.T.shape, W[k-1].shape
                    g = np.dot(g.T, W[k-1])
                    # print g.shape, diag[k-1].shape
                    g = np.dot(g, diag[k-1]).T
                    # print g.shape


        for i in xrange(n_layers):
            grad_W[i] /= float(n)
            grad_W[i] += 2 * lambda_ * W[i]
            grad_b[i] /= float(n)

        # print len(grad_W)
        # print grad_b
        # print grad_W
        return grad_W, grad_b

    # def ComputeGradsNum(self, X, Y, W1, b1, W2, b2, lamda, h):
    #     grad_W = np.zeros((self.out_size,self.in_size))
    #     grad_W = np.reshape(grad_W, (self.out_size,self.in_size))
    #     grad_b = np.zeros(self.out_size)
    #     grad_b = np.reshape(grad_b, (self.out_size,1))

    #     c = self.ComputeCost(X, Y, W, b, lamda)

    #     for i in xrange(self.out_size):
    #         b_try = copy.deepcopy(b)
    #         b_try[i] += h
    #         c2 = self.ComputeCost(X,Y,W,b_try,lamda)
    #         grad_b[i] = (c2-c) / h

    #     numel = self.in_size * self.out_size
    #     for i in xrange(numel):
    #         W_try = copy.deepcopy(W)
    #         r = i // self.in_size
    #         c = i % self.in_size

    #         W_try[r][c] += h
    #         c2 = self.ComputeCost(X, Y, W_try, b, lamda)

    #         grad_W[r][c] = (c2-c) / h

    #     return grad_W, grad_b


    # def ComputeGradsNumSlow(self, X, Y, W1, b1, W2, b2, lamda, h):

    #     grad_W = np.zeros((self.out_size,self.in_size))
    #     grad_W = np.reshape(grad_W, (self.out_size,self.in_size))
    #     grad_b = np.zeros(self.out_size)
    #     grad_b = np.reshape(grad_b, (self.out_size,1))

    #     for i in xrange(self.out_size):
    #         b_try = copy.deepcopy(b)
    #         b_try[i] -= h

    #         c1 = self.ComputeCost(X, Y, W, b_try, lamda)

    #         b_try = copy.deepcopy(b)
    #         b_try[i] += h
    #         c2 = self.ComputeCost(X, Y, W, b_try, lamda)

    #         grad_b[i] = (c2-c1) / (2*h)

    #         print "grad_b[%d]=%f" % (i, grad_b[i])

    #     numel = self.in_size * self.out_size

    #     for i in xrange(numel):
    #         r = i // self.in_size
    #         c = i % self.in_size

    #         W_try = copy.deepcopy(W)
    #         W_try[r][c] -= h
    #         c1 = self.ComputeCost(X, Y, W_try, b, lamda)

    #         W_try = copy.deepcopy(W)
    #         W_try[r][c] += h
    #         c2 = self.ComputeCost(X, Y, W_try, b, lamda)

    #         grad_W[r][c] = (c2-c1) / (2*h)

    #     return grad_W, grad_b





    def Train(self, loss_type = CrossEntropy):
        ep = 0

        X_mean = np.reshape(np.zeros(3072), (3072,1))

        X_train, Y_train = self.reader.next_train_data(10000)
        Y_train = np.array(Y_train).T
        X_train = np.reshape(np.array(X_train),(10000,3072)).T / 256.0 #normaliztion
        for i in xrange(3072):
            X_mean[i] = np.mean(X_train[i])
        X_train = X_train - X_mean

        X_val, Y_val = self.reader.next_train_data(10000)
        Y_val = np.array(Y_val).T
        X_val = np.reshape(np.array(X_val),(10000,3072)).T / 256.0 #normaliztion
        X_val = X_val - X_mean

        X_test, Y_test = self.reader.next_test_data()
        Y_test = np.array(Y_test).T
        X_test = np.reshape(np.array(X_test) , (10000,3072)).T / 256.0
        X_test = X_test - X_mean

        train = []
        validate = []
        test = []

        for ep in xrange(self.max_epoch):
            #print "Epoch %d, " % (ep+1)
            batch_index = 0
            v = []

            for batch_index in xrange(10000//self.batch_size):
                X = np.reshape(X_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size],(3072,self.batch_size))
                #print X.shape
                Y = Y_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size]
                #print Y.shape
                P = self.EvaluateClassifier(X, self.W, self.b)

                grad_W, grad_b = self.ComputeGradients(X, Y, P, self.W, self.b, self.lambda_)


                # The evaluation part for the gradients
                # if (CHECK==True):
                #     print self.b
                #     gWt, gbt = self.ComputeGradsNumSlow(X,Y,self.W,self.b,self.lambda_,1e-8)

                #     D_gW = grad_W - gWt
                #     D_gb = grad_b - gbt

                #     # print grad_b, "\n", gbt, "\n", "D=\n", D_gb
                #     eps = 1e-3

                #     E_gW = np.mean(np.abs(D_gW)) / max(eps, np.mean(np.abs(grad_W)+np.abs(gWt)))
                #     E_gb = np.mean(np.abs(D_gb)) / max(eps, np.mean(np.abs(grad_b)+np.abs(gbt)))

                #     print E_gW, E_gb

                J = self.ComputeCost(X,Y,self.W,self.b,self.lambda_)
                # print "Cost = %f in batch %d" % (J,batch_index)
                if batch_index==0:
                    for i in xrange(self.n_layers):
                        self.W[i] += -self.learning_rate * grad_W[i]
                        self.b[i] += -self.learning_rate * grad_b[i]
                else:
                    for i in xrange(self.n_layers):
                        self.W[i] += -self.learning_rate * grad_W[i] - self.rho * v[2*i]
                        self.b[i] += -self.learning_rate * grad_b[i] - self.rho * v[2*i+1]
                v = []
                for i in xrange(self.n_layers):
                    v.append(self.learning_rate * grad_W[i])
                    v.append(self.learning_rate * grad_b[i])
                # print v
                #print grad[0].shape

                # print "self.b = \n", self.b
                # print self.W,self.b
            tr_acc = self.ComputeAccuracy(X_train,Y_train,self.W,self.b)
            v_acc = self.ComputeAccuracy(X_val,Y_val,self.W,self.b)
            t_acc = self.ComputeAccuracy(X_test,Y_test,self.W,self.b)

            train.append(tr_acc*100)
            validate.append(v_acc*100)
            test.append(t_acc*100)

            print("Epoch %2d, train acc= %4.2f%%, validate acc = %4.2f%%, test acc = %4.2f%%" % (ep+1, tr_acc*100, v_acc*100, t_acc*100))
        t_acc = self.ComputeAccuracy(X_test,Y_test,self.W,self.b)
        print "Final accuracy on test set = %4.2f%%" % (t_acc*100)

        return train, validate, test


class Cifar10DataReader():
    def __init__(self,cifar_folder,onehot=True):
        self.cifar_folder=cifar_folder
        self.onehot=onehot
        self.data_index=1
        self.read_next=True
        self.data_label_train=None
        self.data_label_test=None
        self.batch_index=0

    def unpickle(self,f):
        fo = open(f, 'rb')
        d = cPickle.load(fo)
        fo.close()
        return d

    def next_train_data(self,batch_size=100):
        assert 10000%batch_size==0,"10000%batch_size!=0"
        rdata=None
        rlabel=None
        if self.read_next:
            f=os.path.join(self.cifar_folder,"data_batch_%s"%(self.data_index))
            #print 'read: %s'%f
            dic_train=self.unpickle(f)
            self.data_label_train=zip(dic_train['data'],dic_train['labels'])#label 0~9
            np.random.shuffle(self.data_label_train)

            self.read_next=False
            if self.data_index==5:
                self.data_index=1
            else:
                self.data_index+=1

        if self.batch_index<len(self.data_label_train)//batch_size:
            #print self.batch_index
            datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]
            self.batch_index+=1
            rdata,rlabel=self._decode(datum,self.onehot)
        else:
            self.batch_index=0
            self.read_next=True
            return self.next_train_data(batch_size=batch_size)

        return rdata,rlabel

    def _decode(self,datum,onehot):
        rdata=list();rlabel=list()
        if onehot:
            for d,l in datum:
                rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))
                hot=np.zeros(10)
                hot[int(l)]=1
                rlabel.append(hot)
        else:
            for d,l in datum:
                rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))
                rlabel.append(int(l))
        return list(rdata),list(rlabel)

    def next_test_data(self,batch_size=10000):
        if self.data_label_test is None:
            f=os.path.join(self.cifar_folder,"test_batch")
            print 'read: %s'%f
            dic_test=self.unpickle(f)
            data=dic_test['data']
            labels=dic_test['labels'] # 0~9
            self.data_label_test=zip(data,labels)

        np.random.shuffle(self.data_label_test)
        datum=self.data_label_test[0:batch_size]


        return self._decode(datum,self.onehot)

if __name__=="__main__":
    # Initializing the reader
    dr=Cifar10DataReader(cifar_folder="../cifar-10-batches-py/")
    # Initializing classifier
    c = Classifier(n_layers,m_nodes,initialization,rho,eta,lamda,epoch,reader=dr)

    tr, val, t = c.Train()

    plot(tr,val,t)
