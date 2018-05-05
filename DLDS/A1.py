import cPickle  
import numpy as np  
import os  
import copy

batch_size = 100
eta = 0.01
lamda = 0.001
epoch = 40


CHECK = False

SVM = 0
CrossEntropy = 1

def max(a,b):
    return a if a > b else b

class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, in_size, out_size, eta=0.1, lamda=0, epoch=100, batch_size=batch_size, reader=None):
        super(Classifier, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.learning_rate = eta
        self.lambda_=lamda
        self.max_epoch = epoch
        self.W = np.random.normal(0,0.0001, (out_size,in_size))
        #print self.W
        self.b = np.array([0.0 for i in xrange(out_size)])
        self.b = np.reshape(self.b, (out_size,1))
        #print "b.shape=", self.b.shape
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
        #print n
        P = np.zeros((self.out_size,n))
        #print P.shape
        P = np.reshape(P,(self.out_size,n))
        #print P.shape, P[:,1].shape
        #print type(P[:,1])
        for i in xrange(n):
            s = np.dot(W,np.reshape(X[:,i],(self.in_size,1))) + b
            #print s.shape
            p = self.softmax(s)
            #print p.shape
            if i == 0:
                P = p
            else:
                P = np.hstack([P,p])
        #print P.shape
        return P
    
    def ComputeCost(self, X, Y, W, b, lambda_):
        n = X.shape[1]
        P = self.EvaluateClassifier(X, W, b)
        # print P[0].shape
        L = 0.0
        for i in xrange(n):
            L += -np.log(np.dot(np.reshape(Y[:,i],(1,self.out_size)),np.reshape(P[:,i],(self.out_size,1))))[0][0]
        # print "L = ", L[0][0]
        J = L / float(n) + lambda_ * np.sum(np.square(W))

        # print "cost=%f" % J
        return J
    
    def ComputeAccuracy(self, X, Y, W, b):
        n = X.shape[1]
        #print Y.shape
        P = self.EvaluateClassifier(X, W, b)
        labels = np.array([np.argmax(Y[:,i]) for i in xrange(n)])
        out = np.array([np.argmax(P[:,i]) for i in xrange(n)])
        num = 0.0
        for i in xrange(n):
            if labels[i]==out[i]:
                num += 1.0
        acc = num / float(n)
        return acc
        
    def ComputeGradients(self, X, Y, P, W, lambda_):
        n = X.shape[1]
        #print X.shape, Y.shape
        grad_W = np.zeros((self.out_size,self.in_size))
        grad_W = np.reshape(grad_W, (self.out_size,self.in_size))
        grad_b = np.zeros(self.out_size)
        grad_b = np.reshape(grad_b, (self.out_size,1))
        
        for i in xrange(n):
            g = -(np.reshape(Y[:,i],(self.out_size,1)) - np.reshape(P[:,i],(self.out_size,1)))
            #print g.shape, grad_b.shape
            grad_b += g
            grad_W += np.dot(g,np.reshape(X[:,i],(1,self.in_size)))
        grad_W /= n
        grad_b /= n
        grad_W += 2 * lambda_ * W
        #print grad_b
        return grad_W, grad_b
        
    def ComputeGradsNum(self, X, Y, W, b, lamda, h):
        grad_W = np.zeros((self.out_size,self.in_size))
        grad_W = np.reshape(grad_W, (self.out_size,self.in_size))
        grad_b = np.zeros(self.out_size)
        grad_b = np.reshape(grad_b, (self.out_size,1))
        
        c = self.ComputeCost(X, Y, W, b, lamda)
        
        # print "c = %f" % c

        for i in xrange(self.out_size):
            b_try = copy.deepcopy(b)
            b_try[i] += h
            c2 = self.ComputeCost(X,Y,W,b_try,lamda)
            grad_b[i] = (c2-c) / h
            # if i == self.out_size-1:
            #     print "c2 = %f" % c2
            
        numel = self.in_size * self.out_size
        for i in xrange(numel):
            W_try = copy.deepcopy(W)
            r = i // self.in_size
            c = i % self.in_size
            # if i == numel-1:
            #     print "c2 = %f" % c2

            # print "(r,c)=(%d,%d)\n" % (r,c)

            W_try[r][c] += h
            c2 = self.ComputeCost(X, Y, W_try, b, lamda)
            
            grad_W[r][c] = (c2-c) / h
        
        return grad_W, grad_b


    def ComputeGradsNumSlow(self, X, Y, W, b, lamda, h):

        grad_W = np.zeros((self.out_size,self.in_size))
        grad_W = np.reshape(grad_W, (self.out_size,self.in_size))
        grad_b = np.zeros(self.out_size)
        grad_b = np.reshape(grad_b, (self.out_size,1))
        
        for i in xrange(self.out_size):
            b_try = copy.deepcopy(b)
            b_try[i] -= h
            
            # print b_try, b

            c1 = self.ComputeCost(X, Y, W, b_try, lamda)

            b_try = copy.deepcopy(b)
            b_try[i] += h
            c2 = self.ComputeCost(X, Y, W, b_try, lamda)

            grad_b[i] = (c2-c1) / (2*h)

            print "grad_b[%d]=%f" % (i, grad_b[i])

        numel = self.in_size * self.out_size

        for i in xrange(numel):  
            r = i // self.in_size
            c = i % self.in_size

            W_try = copy.deepcopy(W)
            W_try[r][c] -= h
            c1 = self.ComputeCost(X, Y, W_try, b, lamda)

            W_try = copy.deepcopy(W)
            W_try[r][c] += h
            c2 = self.ComputeCost(X, Y, W_try, b, lamda)

            grad_W[r][c] = (c2-c1) / (2*h)

        return grad_W, grad_b




        
    def Train(self, loss_type = CrossEntropy):
        ep = 0
        
        X_t, Y_t = self.reader.next_test_data()
        
        # Permuting
        X_t = np.reshape(np.array(X_t) / 256.0, (10000,3072)).T
        Y_t = np.array(Y_t).T        

        for ep in xrange(self.max_epoch):
            print "Epoch %d, " % (ep+1)
            batch_index = 0
            for batch_index in xrange(50000//self.batch_size):
                X, Y = self.reader.next_train_data(self.batch_size)
                X = np.array(X) 
                X = np.reshape(X,(self.batch_size,3072)).T / 256.0 #normaliztion
                Y = np.array(Y).T
                
                P = self.EvaluateClassifier(X,self.W,self.b)

                grad_W, grad_b = self.ComputeGradients(X,Y,P,self.W,self.lambda_)


                # The evaluation part for the gradients
                if (CHECK==True):
                    print self.b
                    gWt, gbt = self.ComputeGradsNumSlow(X,Y,self.W,self.b,self.lambda_,1e-8)
                    
                    D_gW = grad_W - gWt
                    D_gb = grad_b - gbt

                    # print grad_b, "\n", gbt, "\n", "D=\n", D_gb
                    eps = 1e-3

                    E_gW = np.mean(np.abs(D_gW)) / max(eps, np.mean(np.abs(grad_W)+np.abs(gWt)))
                    E_gb = np.mean(np.abs(D_gb)) / max(eps, np.mean(np.abs(grad_b)+np.abs(gbt)))
                    
                    print E_gW, E_gb
                
                J = self.ComputeCost(X,Y,self.W,self.b,self.lambda_)
                # print "Cost = %f in batch %d" % (J,batch_index)
                
                self.W += -self.learning_rate * grad_W
                self.b += -self.learning_rate * grad_b
                # print "self.b = \n", self.b
            acc = self.ComputeAccuracy(X_t,Y_t,self.W,self.b)
            print("Accuracy = %f%% after epoch %d" % (acc*100,ep+1))

            
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
            labels=dic_test['labels']#0~9  
            self.data_label_test=zip(data,labels)  
          
        np.random.shuffle(self.data_label_test)  
        datum=self.data_label_test[0:batch_size]  
          
        
        return self._decode(datum,self.onehot)  
        
if __name__=="__main__":  
    # Initializing the reader
    dr=Cifar10DataReader(cifar_folder="cifar-10-batches-py/")  
    
    import matplotlib.pyplot as plt  
    
    # Initializing classifier
    c = Classifier(32*32*3,10,eta,lamda,epoch,reader=dr)
    c.Train()

    # Showing the weight matrix
    s_im = []
    for i in xrange(10):
        im = np.reshape(c.W[i,:],(32,32,3))
        s_im.append((im - np.min(im)) / (np.max(im) - np.min(im)))
        # plt.imshow(s_im[i])

    fig = plt.figure()
    for i in xrange(10):
        ax = fig.add_subplot(2,5,i+1)
        ax.imshow(s_im[i])
    
    plt.show()

