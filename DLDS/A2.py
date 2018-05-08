import cPickle
import numpy as np
import os
import copy
import matplotlib.pyplot as plt


batch_size = 100
eta = 0.05
lamda = 0.005
epoch = 60

out_size = 10
in_size = 3072
m_nodes = 50
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
    def __init__(self, in_size, out_size, m_nodes, rho, eta=0.1, lamda=0, epoch=100, batch_size=batch_size, reader=None):
        super(Classifier, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.m_nodes = m_nodes # number of hidden nodes
        self.rho = rho
        self.learning_rate = eta
        self.lambda_= lamda
        self.max_epoch = epoch

        W1 = np.random.normal(0,0.001, (m_nodes,in_size))
        W2 = np.random.normal(0,0.001, (out_size,m_nodes))
        self.W = [[],W1,W2]

        b1 = np.reshape(np.array([0.0 for i in xrange(m_nodes)]), (m_nodes,1))
        b2 = np.reshape(np.array([0.0 for i in xrange(out_size)]), (out_size,1))
        self.b = [[],b1,b2]

        self.batch_size = batch_size
        # self.RW = np.sum(np.square(self.W)) * self.lambda_
        self.reader = reader

    def softmax(self,s):
        s = np.exp(s)
        a = np.sum(s)
        s = s / a
        return s

    def EvaluateClassifier(self, X, W, b):
        n = X.shape[1]
        P = np.zeros((self.out_size,n))
        P = np.reshape(P,(self.out_size,n))

        for i in xrange(n):
            s1 = np.dot(W[1],np.reshape(X[:,i],(self.in_size,1))) + b[1]
            h = np.abs((np.abs(s1)+s1) / 2) # max(0,s1)

            s = np.dot(W[2],np.reshape(h,(self.m_nodes,1))) + b[2]
            p = self.softmax(s)
            if i == 0:
                P = p
            else:
                P = np.hstack([P,p])
        return P

    def ComputeCost(self, X, Y, W, b, lambda_):
        n = X.shape[1]
        P = self.EvaluateClassifier(X, W, b)
        L = 0.0
        for i in xrange(n):
            L += -np.log(np.dot(np.reshape(Y[:,i],(1,self.out_size)),np.reshape(P[:,i],(self.out_size,1))))[0][0]
        J = L / float(n) + lambda_ * (np.sum(np.square(W[1])) + np.sum(np.square(W[2])))
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
        grad_W1 = np.zeros((self.m_nodes,self.in_size))
        grad_W1 = np.reshape(grad_W1, (self.m_nodes,self.in_size))
        grad_b1 = np.zeros(self.m_nodes)
        grad_b1 = np.reshape(grad_b1, (self.m_nodes,1))
        grad_W2 = np.zeros((self.out_size,self.m_nodes))
        grad_W2 = np.reshape(grad_W2, (self.out_size,self.m_nodes))
        grad_b2 = np.zeros(self.out_size)
        grad_b2 = np.reshape(grad_b2, (self.out_size,1))

        for i in xrange(n):
            s1 = np.dot(W[1],np.reshape(X[:,i],(self.in_size,1))) + b[1]
            h = np.abs((np.abs(s1)+s1) / 2) # max(0,s1)

            diag = np.zeros((self.m_nodes,self.m_nodes))
            diag = np.reshape(diag, (self.m_nodes,self.m_nodes))

            for j in xrange(self.m_nodes):
                if s1[j] > 0:
                    diag[j][j] = 1

            g = -(np.reshape(Y[:,i],(self.out_size,1)) - np.reshape(P[:,i],(self.out_size,1)))

            grad_b2 += g
            grad_W2 += np.dot(g,np.reshape(h,(1,self.m_nodes)))

            g = np.dot(g.T, W[2])
            g = np.dot(g, diag).T

            grad_b1 += g
            grad_W1 += np.dot(g,np.reshape(X[:,i],(1,self.in_size)))

        grad_W1 /= n
        grad_b1 /= n
        grad_W2 /= n
        grad_b2 /= n
        grad_W1 += 2 * lambda_ * W[1]
        grad_W2 += 2 * lambda_ * W[2]
        #print grad_b
        return grad_W1, grad_b1, grad_W2, grad_b2

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

            for batch_index in xrange(10000//self.batch_size):
                X = np.reshape(X_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size],(3072,self.batch_size))
                #print X.shape
                Y = Y_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size]
                #print Y.shape
                P = self.EvaluateClassifier(X, self.W, self.b)

                grad_W1, grad_b1, grad_W2, grad_b2 = self.ComputeGradients(X, Y, P, self.W, self.b, self.lambda_)


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
                    self.W[1] += -self.learning_rate * grad_W1
                    self.b[1] += -self.learning_rate * grad_b1
                    self.W[2] += -self.learning_rate * grad_W2
                    self.b[2] += -self.learning_rate * grad_b2
                else:
                    self.W[1] += -self.learning_rate * grad_W1 - self.rho * v[0]
                    self.b[1] += -self.learning_rate * grad_b1 - self.rho * v[1]
                    self.W[2] += -self.learning_rate * grad_W2 - self.rho * v[2]
                    self.b[2] += -self.learning_rate * grad_b2 - self.rho * v[3]
                v = [self.learning_rate * grad_W1,
                     self.learning_rate * grad_b1,
                     self.learning_rate * grad_W2,
                     self.learning_rate * grad_b2]
                #print grad[0].shape

                # print "self.b = \n", self.b
            tr_acc = self.ComputeAccuracy(X_train,Y_train,self.W,self.b)
            v_acc = self.ComputeAccuracy(X_val,Y_val,self.W,self.b)
            t_acc = self.ComputeAccuracy(X_test,Y_test,self.W,self.b)

            train.append(tr_acc*100)
            validate.append(v_acc*100)
            test.append(t_acc*100)

            print("Epcoh %2d, train acc= %4.2f%%, validate acc = %4.2f%%, test acc = %4.2f%%" % (ep+1, tr_acc*100, v_acc*100, t_acc*100))
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
    dr=Cifar10DataReader(cifar_folder="cifar-10-batches-py/")
    # Initializing classifier
    c = Classifier(in_size,out_size,m_nodes,rho,eta,lamda,epoch,reader=dr)

    tr, val, t = c.Train()

    plot(tr,val,t)
