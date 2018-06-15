import cPickle  
import numpy as np  
import os  
import copy
import matplotlib.pyplot as plt  
    

batch_size = 100
eta = 0.021
lamda = 0.0003
epoch = 30

m_nodes = 50
rho = 0.95

# initialization = 'He'
initialization = 'Xavier'

CHECK = False

SVM = 0
CrossEntropy = 1

def max(a,b):
    return a if a > b else b
    
def plot_acc(a,b,c):
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

def plot_cost(tr,val):
    ep = []
    for i in xrange(len(tr)):
        ep.append(i)
    
    l1 = plt.plot(ep,tr,'r',label='train')
    l2 = plt.plot(ep,val,'b',label='valadation')
    plt.title('The training process')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
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
        sigma = []
        if initialization == 'Xavier':
            sigma.append(1 / np.sqrt(float(in_size)))
            sigma.append(1 / np.sqrt(float(m_nodes)))
        else:
            if initialization == 'He':
                sigma.append(np.sqrt(2 / float(in_size)))
                sigma.append(np.sqrt(2 / float(m_nodes)))
            else:
                sigma.append(0.001)
                sigma.append(0.001)
        self.W1 = np.random.normal(0,sigma[0], (m_nodes,in_size))
        self.W2 = np.random.normal(0,sigma[1], (out_size,m_nodes))

        self.b1 = np.array([0.0 for i in xrange(m_nodes)])
        self.b1 = np.reshape(self.b1, (m_nodes,1))

        self.b2 = np.array([0.0 for i in xrange(out_size)])
        self.b2 = np.reshape(self.b2, (out_size,1))

        self.batch_size = batch_size
        # self.RW = np.sum(np.square(self.W)) * self.lambda_
        self.reader = reader

    def softmax(self,s):
        s = np.exp(s)
        a = np.sum(s)
        s = s / a
        return s
    
    def EvaluateClassifier(self, X, W1, b1, W2, b2):
        n = X.shape[1]
        #print n
        P = np.zeros((self.out_size,n))
        #print P.shape
        P = np.reshape(P,(self.out_size,n))
        #print P.shape, P[:,1].shape
        #print type(P[:,1])

        for i in xrange(n):
            s1 = np.dot(W1,np.reshape(X[:,i],(self.in_size,1))) + b1
            h = np.abs((np.abs(s1)+s1) / 2) # max(0,s1)

            s = np.dot(W2,np.reshape(h,(self.m_nodes,1))) + b2
            #print s.shape
            p = self.softmax(s)
            #print p.shape
            if i == 0:
                P = p
            else:
                P = np.hstack([P,p])
        #print P.shape
        return P
    
    def ComputeCost(self, X, Y, W1, b1, W2, b2, lambda_):
        n = X.shape[1]
        P = self.EvaluateClassifier(X, W1, b1, W2, b2)
        # print P[0].shape
        L = 0.0
        for i in xrange(n):
            L += -np.log(np.dot(np.reshape(Y[:,i],(1,self.out_size)),np.reshape(P[:,i],(self.out_size,1))))[0][0]
        # print "L = ", L[0][0]
        J = L / float(n) + lambda_ * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        # print "cost=%f" % J
        return J
    
    def ComputeAccuracy(self, X, Y, W1, b1, W2, b2):
        n = X.shape[1]
        #print Y.shape
        P = self.EvaluateClassifier(X, W1, b1, W2, b2)
        labels = np.array([np.argmax(Y[:,i]) for i in xrange(n)])
        out = np.array([np.argmax(P[:,i]) for i in xrange(n)])
        num = 0.0
        for i in xrange(n):
            if labels[i]==out[i]:
                num += 1.0
        acc = num / float(n)
        return acc
        
    def ComputeGradients(self, X, Y, P, W1, b1, W2, lambda_):
        n = X.shape[1]
        #print X.shape, Y.shape
        grad_W1 = np.zeros((self.m_nodes,self.in_size))
        grad_W1 = np.reshape(grad_W1, (self.m_nodes,self.in_size))
        grad_b1 = np.zeros(self.m_nodes)
        grad_b1 = np.reshape(grad_b1, (self.m_nodes,1))
        grad_W2 = np.zeros((self.out_size,self.m_nodes))
        grad_W2 = np.reshape(grad_W2, (self.out_size,self.m_nodes))
        grad_b2 = np.zeros(self.out_size)
        grad_b2 = np.reshape(grad_b2, (self.out_size,1))
        
        for i in xrange(n):
            s1 = np.dot(W1,np.reshape(X[:,i],(self.in_size,1))) + b1
            h = np.abs((np.abs(s1)+s1) / 2) # max(0,s1)
            
            diag = np.diag(np.reshape(s1, (s1.size,)))
            for j in xrange(s1.size):
                diag[j][j] = 1 if diag[j][j] > 0 else 0
            # diag = np.reshape(diag, (self.m_nodes,self.m_nodes))
            
            
            g = -(np.reshape(Y[:,i],(self.out_size,1)) - np.reshape(P[:,i],(self.out_size,1)))
            #print g.shape, grad_b.shape
            grad_b2 += g
            grad_W2 += np.dot(g,np.reshape(h,(1,self.m_nodes)))
            
            g = np.dot(g.T, W2)
            g = np.dot(g, diag).T
            
            grad_b1 += g
            grad_W1 += np.dot(g,np.reshape(X[:,i],(1,self.in_size)))
            
        grad_W1 /= n
        grad_b1 /= n
        grad_W2 /= n
        grad_b2 /= n
        grad_W1 += 2 * lambda_ * W1
        grad_W2 += 2 * lambda_ * W2
        #print grad_b
        return grad_W1, grad_b1, grad_W2, grad_b2
        
    def ComputeGradsNum(self, X, Y, W1, b1, W2, b2, lamda, h):
        grad_W1 = np.reshape(np.zeros(W1.shape),W1.shape)
        grad_W2 = np.reshape(np.zeros(W2.shape),W2.shape)
        grad_b1 = np.reshape(np.zeros(b1.shape),b1.shape)
        grad_b2 = np.reshape(np.zeros(b2.shape),b2.shape)
        # print grad_W1.shape, grad_W2.shape, grad_b1.shape, grad_b2.shape
        
        c0 = self.ComputeCost(X, Y, W1, b1, W2, b2, lamda)

        # b1
        print "Computing grad_b1..."
        for i in xrange(b1.size):
            b_try = copy.deepcopy(b1)
            b_try[i] += h
            c2 = self.ComputeCost(X,Y,W1,b_try,W2,b2,lamda)
            grad_b1[i] = (c2-c0) / h

        # b2
        print "Computing grad_b2..."
        for i in xrange(b2.size):
            b_try = copy.deepcopy(b2)
            b_try[i] += h
            c2 = self.ComputeCost(X,Y,W1,b1,W2,b_try,lamda)
            grad_b2[i] = (c2-c0) / h
        
        # W1
        print "Computing grad_W1..."
        for i in xrange(W1.size):
            W_try = copy.deepcopy(W1)
            r = i // W1.shape[1]
            c = i % W1.shape[1]

            W_try[r][c] += h
            c2 = self.ComputeCost(X, Y, W_try, b1, W2, b2, lamda)
            grad_W1[r][c] = (c2-c0) / h

        # W2
        print "Computing grad_W2..."
        for i in xrange(W2.size):
            W_try = copy.deepcopy(W2)
            r = i // W2.shape[1]
            c = i % W2.shape[1]

            W_try[r][c] += h
            c2 = self.ComputeCost(X, Y, W1, b1, W_try, b2, lamda)
            
            grad_W2[r][c] = (c2-c0) / h
        
        return grad_W1, grad_b1, grad_W2, grad_b2

        
    def Train(self, loss_type = CrossEntropy):
        ep = 0
        
        X_mean = np.reshape(np.zeros(3072), (3072,1))
      
        for i in range(5):
            X_tr, Y_tr = self.reader.next_train_data(10000)
            Y_tr = np.array(Y_tr).T
            X_tr = np.reshape(np.array(X_tr),(10000,3072)).T / 255.0 #normaliztion
            if i == 0:
                X_train = X_tr.copy()
                Y_train = Y_tr.copy()
            else:
                X_train = np.hstack([X_train,X_tr])
                Y_train = np.hstack([Y_train,Y_tr])

        for i in xrange(3072):
            X_mean[i] = np.mean(X_train[i])
        X_train = X_train - X_mean

        X_val = np.array(X_train[:,49000: ])
        Y_val = np.array(Y_train[:,49000: ])
        
        X_test, Y_test = self.reader.next_test_data()
        Y_test = np.array(Y_test).T
        X_test = np.reshape(np.array(X_test) , (10000,3072)).T / 255.0
        X_test = X_test - X_mean

        train = []
        validate = []
        test = []

        t_c = []
        v_c = []

        J_tr = J = self.ComputeCost(X_train,Y_train,self.W1,self.b1,self.W2,self.b2,self.lambda_)
        J_val = self.ComputeCost(X_val,Y_val,self.W1,self.b1,self.W2,self.b2,self.lambda_)    
        t_c.append(J_tr)
        v_c.append(J_val)
        
        for ep in xrange(self.max_epoch):
            #print "Epoch %d, " % (ep+1)
            batch_index = 0
            
            for batch_index in xrange(10000//self.batch_size):
                X = np.reshape(X_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size],(3072,self.batch_size))
                #print X.shape
                Y = Y_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size]
                #print Y.shape
                P = self.EvaluateClassifier(X, self.W1, self.b1, self.W2, self.b2)

                grad_W1, grad_b1, grad_W2, grad_b2 = self.ComputeGradients(X, Y, P, self.W1, self.b1, self.W2, self.lambda_)
                

                # The evaluation part for the gradients
                if (CHECK==True):

                    gW1t, gb1t, gW2t, gb2t = self.ComputeGradsNum(X,Y,self.W1,self.b1,self.W2,self.b2,self.lambda_,1e-5)
                    
                    D_gW1 = grad_W1 - gW1t
                    D_gb1 = grad_b1 - gb1t
                    D_gW2 = grad_W2 - gW2t
                    D_gb2 = grad_b2 - gb2t

                    # print grad_b, "\n", gbt, "\n", "D=\n", D_gb
                    eps = 1e-6
                    # print gW1t, grad_W1

                    E_gW1 = np.mean(np.abs(D_gW1)) / max(eps, np.mean(np.abs(grad_W1)+np.abs(gW1t)))
                    E_gb1 = np.mean(np.abs(D_gb1)) / max(eps, np.mean(np.abs(grad_b1)+np.abs(gb1t)))
                    E_gW2 = np.mean(np.abs(D_gW2)) / max(eps, np.mean(np.abs(grad_W2)+np.abs(gW2t)))
                    E_gb2 = np.mean(np.abs(D_gb2)) / max(eps, np.mean(np.abs(grad_b2)+np.abs(gb2t)))

                    
                    print E_gW1, E_gb1, E_gW2, E_gb2
                
                # J = self.ComputeCost(X,Y,self.W1,self.b1,self.W2,self.b2,self.lambda_)
                # print "Cost = %f in batch %d" % (J,batch_index)
                if batch_index==0:
                    self.W1 += -self.learning_rate * grad_W1
                    self.b1 += -self.learning_rate * grad_b1
                    self.W2 += -self.learning_rate * grad_W2
                    self.b2 += -self.learning_rate * grad_b2
                else:
                    self.W1 += -self.learning_rate * grad_W1 - self.rho * v[0]
                    self.b1 += -self.learning_rate * grad_b1 - self.rho * v[1]
                    self.W2 += -self.learning_rate * grad_W2 - self.rho * v[2]
                    self.b2 += -self.learning_rate * grad_b2 - self.rho * v[3]
                v = [self.learning_rate * grad_W1, 
                     self.learning_rate * grad_b1, 
                     self.learning_rate * grad_W2, 
                     self.learning_rate * grad_b2]
                #print grad[0].shape
                    
                # print "self.b = \n", self.b
            tr_acc = self.ComputeAccuracy(X_train,Y_train,self.W1,self.b1,self.W2,self.b2)
            v_acc = self.ComputeAccuracy(X_val,Y_val,self.W1,self.b1,self.W2,self.b2)
            t_acc = self.ComputeAccuracy(X_test,Y_test,self.W1,self.b1,self.W2,self.b2)
            
            train.append(tr_acc*100)
            validate.append(v_acc*100)
            test.append(t_acc*100)

            J_tr = J = self.ComputeCost(X_train,Y_train,self.W1,self.b1,self.W2,self.b2,self.lambda_)
            J_val = self.ComputeCost(X_val,Y_val,self.W1,self.b1,self.W2,self.b2,self.lambda_)
            
            t_c.append(J_tr)
            v_c.append(J_val)

            if J_tr > 3 * t_c[0]:
                print "Cost exploded, end training"
                break

            print("Epoch %2d, train acc= %4.2f%%, validate acc = %4.2f%%, test acc = %4.2f%%" % (ep+1, tr_acc*100, v_acc*100, t_acc*100))
        t_acc = self.ComputeAccuracy(X_test,Y_test,self.W1,self.b1,self.W2,self.b2)
        print "Final accuracy on test set = %4.2f%%" % (t_acc*100)
        
        return train, validate, test, t_c, v_c

            
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
    dr=Cifar10DataReader(cifar_folder="../cifar-10-batches-py/")  
    

    # Initializing classifier
    c = Classifier(32*32*3,10,m_nodes,rho,eta,lamda,epoch,reader=dr)
    
    tr, val, t, tr_cost, val_cost = c.Train()

    plot_acc(tr,val,t)
    plot_cost(tr_cost,val_cost)

    # f = open("fine_search.txt", "w")

    # e_min = np.log10(0.01)
    # e_max = np.log10(0.05)

    # result = np.array([0,0,0])
    # for i in xrange(75):
    #   e = e_min + (e_max - e_min) * np.random.random(1)[0]
    #   eta = np.power(10, e)
    #   lamda = 0.0001 + (0.015 - 0.0001) * np.random.random(1)[0]
    #   epoch = 20
    #   c = Classifier(32*32*3,10,m_nodes,rho,eta,lamda,epoch,reader=dr)
    #   tr, val, t, tr_cost, val_cost = c.Train()
    #   f.write("eta = %f, lambda = %f, test_acc after %d epochs is %4.2f%%\n" % (eta, lamda, epoch, t[-1]))
    #   result = np.vstack([result, np.array([t[-1],eta,lamda])])

    # ind = np.argsort(-result, axis=0)
    # index = ind[:,0]
    # best_ten = result[index[:10]]
    # f.write("\nThe best ten results are: (acc eta lambda)\n")
    # for i in xrange(10):
    #   f.write("%4.2f%%  %6f  %6f\n" % (best_five[i,0],best_five[i,1],best_five[i,2]))

    # f.close()

