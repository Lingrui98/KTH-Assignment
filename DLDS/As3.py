import cPickle
import numpy as np
import os
import copy
import matplotlib.pyplot as plt

BN = True

batch_size = 100
eta = 0.01
lamda = 0.0001
epoch = 20

# initialization = 'He'
initialization = 'Xavier'
# initialization = 0

n_layers = 3

out_size = 10
in_size = 3072
m_nodes = [in_size,50,30,out_size]
rho = 0.9

CHECK = False

SVM = 0
CrossEntropy = 1

def max(a,b):
    return a if a > b else b

def plot_accuracy(a,b,c):
    ep = []
    for i in xrange(len(a)):
        ep.append(i+1)

    l1 = plt.plot(ep,a,'r',label='train')
    l2 = plt.plot(ep,b,'g',label='validate')
    l3 = plt.plot(ep,c,'b',label='test')
    plt.title('Accuracy during the training process')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.legend(loc='best')
    plt.show()

def plot_cost(tr,val):
    ep = []
    for i in xrange(len(tr)):
        ep.append(i+1)
    train    = plt.plot(ep,tr, 'r',label='train')
    validate = plt.plot(ep,val,'b',label='validate')
    plt.title('Cost during the training process')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend(loc='best')
    plt.show()


class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, n_layers, m_nodes, init, rho, BN, eta=0.1, lamda=0, epoch=20, batch_size=batch_size, reader=None):
        super(Classifier, self).__init__()
        self.out_size = m_nodes[-1]
        self.in_size = m_nodes[0]
        self.n_layers = n_layers
        self.m_nodes = m_nodes # number of hidden nodes
        self.rho = rho
        self.learning_rate = eta
        self.lambda_= lamda
        self.max_epoch = epoch
        self.BN = BN
        self.alpha = 0.99

        self.mu_av  = [np.reshape(np.zeros(m_nodes[i+1]), (m_nodes[i+1],1)) for i in xrange(n_layers)]
        self.var_av = [np.reshape(np.zeros(m_nodes[i+1]), (m_nodes[i+1],1)) for i in xrange(n_layers)]

        self.set = False # indicates whether mu_av and var_av are set

        # len(W) equals to the number of layers
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
        # self.reader = reader

    def softmax(self,s):
        s = np.exp(s)
        a = np.sum(s)
        s = s / a
        return s

    def BatchNormalize(self,s,mu,var,e): # returns the normalized s_hat for ONE layer in ONE sample
        m = s.shape[0] # number of nodes in this layer
        M = np.reshape(np.zeros(m*m),(m,m))
        for i in xrange(m):
            M[i][i] = 1 / np.sqrt(var[i] + e)
        s_hat = np.dot(M,s-mu)
        return s_hat

# Forward pass
    def EvaluateClassifier(self, X, W, b, update=False, test=False, mu_a=None, var_a=None):
        n = X.shape[1]
        P = np.zeros((self.out_size,n))
        P = np.reshape(P,(self.out_size,n))

        s_rec = [[np.reshape(X[:,i],(self.in_size,1))] for i in xrange(n)] # record of un-normalized scores
        s_hat = [[] for i in xrange(n)] # normalized scores
        x = [[s_rec[i][0]] for i in xrange(n)]

        mu = []
        var = []


        for i in xrange(self.n_layers):
            s = np.reshape(np.zeros(self.m_nodes[i+1]), (self.m_nodes[i+1],1))
            v = np.reshape(np.zeros(self.m_nodes[i+1]), (self.m_nodes[i+1],1))
            for j in xrange(n):
                s_rec[j].append(np.dot(W[i],x[j][i]) + b[i])
                if self.BN == True:
                    s += np.array(s_rec[j][i+1])
            if self.BN == True:
                mu.append(s / float(n))

                for j in xrange(n):
                    v += np.square(np.array(s_rec[j][i+1])-mu[i])
                var.append(v / float(n))

                for j in xrange(n):
                    if test == True:
                        if mu_a==None or var_a==None:
                            s_hat[j].append(self.BatchNormalize(s_rec[j][i+1],self.mu_av[i],self.var_av[i],1e-5))
                        else:
                            s_hat[j].append(self.BatchNormalize(s_rec[j][i+1],mu_a[i],var_a[i],1e-5))
                    else:
                        s_hat[j].append(self.BatchNormalize(s_rec[j][i+1],mu[i],var[i],1e-5))
                    x[j].append(np.abs((np.abs(s_hat[j][i])+s_hat[j][i]) / 2))
            else:
                for j in xrange(n):
                    x[j].append(np.abs((np.abs(s_rec[j][i+1])+s_rec[j][i+1]) / 2))

        for i in xrange(n):
            p = self.softmax(s_rec[i][-1])
            if i == 0:
                P = p
            else:
                P = np.hstack([P,p])

        if update == True:
            if self.BN == True:
                for i in xrange(self.n_layers):
                    if self.set == False:
                        self.mu_av[i]  =  mu[i]
                        self.var_av[i] = var[i]
                        self.set = True
                    else:
                        self.mu_av[i]  = self.alpha *  self.mu_av[i] + (1 - self.alpha) *  mu[i]
                        self.var_av[i] = self.alpha * self.var_av[i] + (1 - self.alpha) * var[i]
            return P, s_hat, s_rec, x
        else:
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

    def ComputeAccuracy(self, X, Y, W, b, test=False, mu=None, var=None):
        n = X.shape[1]
        if mu==None or var==None:
            P = self.EvaluateClassifier(X, W, b,test=test)
        else:
            P = self.EvaluateClassifier(X, W, b,test=test,mu_a=mu,var_a=var)
        labels = np.array([np.argmax(Y[:,i]) for i in xrange(n)])
        out = np.array([np.argmax(P[:,i]) for i in xrange(n)])
        num = 0.0
        for i in xrange(n):
            if labels[i]==out[i]:
                num += 1.0
        acc = num / float(n)
        return acc

    def ComputeGradients(self, X, Y, W, b, lambda_):

        P , s_hat, s_rec, x = self.EvaluateClassifier(X, self.W, self.b, update=True)


        n = X.shape[1]
        grad_W = []
        grad_b = []
        for i in xrange(self.n_layers):
            grad_W.append(np.reshape(np.zeros((self.m_nodes[i+1],self.m_nodes[i])), (self.m_nodes[i+1],self.m_nodes[i])))
            grad_b.append(np.reshape(np.zeros(self.m_nodes[i+1]), (self.m_nodes[i+1],1)))

        diag = [[]]


        for i in xrange(n):
            if self.BN == False:
                s = s_rec[i]
            else:
                s = [[]]+s_hat[i]

            for j in xrange(self.n_layers):
                if j < self.n_layers-1:
                    diag.append(np.reshape(np.zeros((self.m_nodes[j+1],self.m_nodes[j+1])), (self.m_nodes[j+1],self.m_nodes[j+1])))

                    for k in xrange(self.m_nodes[j+1]):
                        if s[j+1][k] > 0:
                            (diag[j+1])[k][k] = 1

            g = -(np.reshape(Y[:,i],(self.out_size,1)) - np.reshape(P[:,i],(self.out_size,1)))

            for j in xrange(self.n_layers):
                k = self.n_layers - j

                grad_b[k-1] += g
                grad_W[k-1] += np.dot(g,np.reshape(x[i][k-1],(1,self.m_nodes[k-1])))

                if k > 1:
                    g = np.dot(g.T, W[k-1])
                    g = np.dot(g, diag[k-1]).T



        for i in xrange(n_layers):
            grad_W[i] /= float(n)
            grad_W[i] += 2 * lambda_ * W[i]
            grad_b[i] /= float(n)

        return grad_W, grad_b

    def ComputeGradsNum(self, X, Y, W, b, lamda, h):
        grad_W = []
        grad_b = []
        for i in xrange(self.n_layers):
            grad_W.append(np.reshape(np.zeros((self.m_nodes[i+1],self.m_nodes[i])), (self.m_nodes[i+1],self.m_nodes[i])))
            grad_b.append(np.reshape(np.zeros(self.m_nodes[i+1]), (self.m_nodes[i+1],1)))

        c0 = self.ComputeCost(X, Y, W, b, lamda)

        for i in xrange(len(b)):
            for j in xrange(b[i].size):
                b_try = copy.deepcopy(b)
                b_try[i][j] += h
                c2 = self.ComputeCost(X,Y,W,b_try,lamda)
                grad_b[i][j] = (c2-c0) / h

        for i in xrange(len(W)):
            for j in xrange(W[i].size):
                W_try = copy.deepcopy(W)
                r = j // W[i].shape[1]
                c = j % W[i].shape[1]

                W_try[i][r][c] += h
                c2 = self.ComputeCost(X, Y, W_try, b, lamda)

                grad_W[i][r][c] = (c2-c0) / h


        return grad_W, grad_b

    def print_pram(self):
        s = 'with' if self.BN == True else 'without'
        print "\nTraining a %d-layer network %s batch normalization..." % (self.n_layers,s)
        print "Using %s initialization\n" % (initialization)
        print "The structure is ", self.m_nodes
        print "initial eta = %3.2f" % (self.learning_rate)
        print "lambda = %5.4f" % (self.lambda_)
        print "max_epoch = %d" % (self.max_epoch)
        print "rho = %3.2f" % (self.rho)
        print "alpha = %3.2f\n" % (self.alpha)


    def Train(self, loss_type = CrossEntropy):
        ep = 0
        self.print_pram()

        train = []
        validate = []
        test = []
        cost = []

        best_W = [[] for i in xrange(self.n_layers)]
        best_b = [[] for i in xrange(self.n_layers)]
        best_mu = [[] for i in xrange(self.n_layers)]
        best_var = [[] for i in xrange(self.n_layers)]
        max_val_acc = 0.0

        eta = self.learning_rate

        for ep in xrange(self.max_epoch):
            batch_index = 0

            v = [] # the momentum

            # if (ep+1) % 1 == 0: # decline of learning rate
            #     eta *= 1.0

            # Mini-batch training in one epoch
            for batch_index in xrange(10000//self.batch_size):
                X = np.reshape(X_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size],(3072,self.batch_size))
                Y = Y_train[:,batch_index*self.batch_size:(batch_index+1)*self.batch_size]

                grad_W, grad_b = self.ComputeGradients(X, Y, self.W, self.b, self.lambda_)

                # The evaluation part for the gradients
                if (CHECK==True):
                    print "Checking gradients..."
                    gWt, gbt = self.ComputeGradsNum(X,Y,self.W,self.b,self.lambda_,1e-5)

                    D_gW = []
                    D_gb = []
                    for i in xrange(self.n_layers):
                        D_gW.append(grad_W[i] - gWt[i])
                        D_gb.append(grad_b[i] - gbt[i])

                    eps = 1e-5
                    E_gW = []
                    E_gb = []
                    for i in xrange(self.n_layers):
                        E_gW.append(np.mean(np.abs(D_gW[i])) / max(eps, np.mean(np.abs(grad_W[i])+np.abs(gWt[i]))))
                        E_gb.append(np.mean(np.abs(D_gb[i])) / max(eps, np.mean(np.abs(grad_b[i])+np.abs(gbt[i]))))
                    print E_gW, E_gb

                # Updating the gradient
                if batch_index==0:
                    for i in xrange(self.n_layers):
                        self.W[i] += -eta * grad_W[i]
                        self.b[i] += -eta * grad_b[i]
                else:
                    for i in xrange(self.n_layers):
                        self.W[i] += -eta * grad_W[i] - self.rho * v[2*i]
                        self.b[i] += -eta * grad_b[i] - self.rho * v[2*i+1]
                v = []
                for i in xrange(self.n_layers):
                    v.append(eta* grad_W[i])
                    v.append(eta * grad_b[i])

            # Compute the cost
            J_tr = self.ComputeCost(X_train,Y_train,self.W,self.b,self.lambda_)
            J_val = self.ComputeCost(X_val,Y_val,self.W,self.b,self.lambda_)
            cost.append([J_tr,J_val])

            # Compute the accuracy
            tr_acc = self.ComputeAccuracy(X_train,Y_train,self.W,self.b)
            v_acc = self.ComputeAccuracy(X_val,Y_val,self.W,self.b,test=True)
            t_acc = self.ComputeAccuracy(X_test,Y_test,self.W,self.b,test=True)
            train.append(tr_acc*100)
            validate.append(v_acc*100)
            test.append(t_acc*100)

            # Keep a record of the best result
            if v_acc > max_val_acc:
                max_val_acc = v_acc
                for i in xrange(self.n_layers):
                    best_W[i] = self.W[i].copy()
                    best_b[i] = self.b[i].copy()
                    best_mu[i] = self.mu_av[i].copy()
                    best_var[i] = self.var_av[i].copy()


            print("Epoch %2d with eta = %4.3f, train acc= %4.2f%%, validate acc = %4.2f%%, test acc = %4.2f%%" % (ep+1, eta, tr_acc*100, v_acc*100, t_acc*100))

        t_acc = test[-1]
        best_result = self.ComputeAccuracy(X_test,Y_test,best_W,best_b,test=True,mu=best_mu,var=best_var) * 100
        print "Final accuracy on test set = %4.2f%%, theoretical best result during training is %4.2f%%" % (t_acc,best_result)

        if self.BN == True:
            best = [best_result, best_W, best_b, best_mu, best_var]
        else:
            best = [best_result, best_W, best_b]

        return train, validate, test, cost, best


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
    X_mean = np.reshape(np.zeros(3072), (3072,1))

    print "Loading training data..."
    X_train, Y_train = dr.next_train_data(10000)
    Y_train = np.array(Y_train).T
    X_train = np.reshape(np.array(X_train),(10000,3072)).T / 256.0 #normaliztion
    for i in xrange(3072):
        X_mean[i] = np.mean(X_train[i])
    X_train = X_train - X_mean

    print "Loading validation data..."
    X_val, Y_val = dr.next_train_data(10000)
    Y_val = np.array(Y_val).T
    X_val = np.reshape(np.array(X_val),(10000,3072)).T / 256.0 #normaliztion
    X_val = X_val - X_mean

    print "Loading test data..."
    X_test, Y_test = dr.next_test_data()
    Y_test = np.array(Y_test).T
    X_test = np.reshape(np.array(X_test) , (10000,3072)).T / 256.0
    X_test = X_test - X_mean

    print "Data loaded..."

    # c = Classifier(n_layers,m_nodes,initialization,rho,BN,eta,lamda,epoch,reader=dr)

    best_arr = [[[0,0],0] for i in xrange(3)] # [[eta,lambda], best_accuracy]
    print best_arr

    for i in xrange(5):
        lr = 0.1 / float((i+1)*(i+1)) 
        for j in xrange(5):
            ld = j*0.002 # [0,0.01]
            a = Classifier(n_layers,m_nodes,initialization,rho,BN,lr,ld,epoch,reader=dr)
            tr, val, t, cost, best = a.Train()
            if best[0] > best_arr[0][1]:
                # print best_arr
                del best_arr[-1]
                best_arr = [[[lr,ld], best[0]]] + best_arr
                # print best_arr

    print "best accuracy achieved with eta = %f, lambda = %f, best_accuracy = %4.2f%%" % (best_arr[0][0][0], best_arr[0][0][1], best_arr[0][1])
    print "best_arr is" , best_arr


    # tr, val, t, cost, best = c.Train()

    # plot_accuracy(tr,val,t)
    # plot_cost(np.array(cost)[:,0],np.array(cost)[:,1])
