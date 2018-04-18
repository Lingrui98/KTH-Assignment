import cPickle  
import numpy as np  
import os  

batch_size = 100
eta = 0.01
lamda = 0
epoch = 40


SVM = 0
CrossEntropy = 1

class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, in_size, out_size, eta=0.1, lamda=0.1, epoch=100, batch_size=batch_size, reader=None):
        super(Classifier, self).__init__()
        self.learning_rate = eta
        self.lambda_=lamda
        self.max_epoch = epoch
        self.W = np.random.normal(0,0.0001, (out_size,in_size))
        #print self.W
        self.b = np.array([0.0 for i in xrange(batch_size*out_size)])
        self.b = np.reshape(self.b, (batch_size,out_size)).T
        #print "b.shape=", self.b.shape

        self.batch_size = batch_size
        # self.RW = np.sum(np.square(self.W)) * self.lambda_
        self.reader = reader

    def softmax(self, s): #Checked
        #print s
        s = np.exp(s)
        #print "s after exp=", s
        #print np.sum(s)
        a = np.sum(s)
        s = s / a
        #print np.sum(s)
        #if np.sum(s)>1 or np.sum(s)<0:
           # print "ERROR"
        #print s
        return s


    def loss(self, inputs, labels, batch_size, typ=CrossEntropy):
        #print inputs,self.W,self.b
        z = np.dot(self.W,inputs) + self.b
#        print z[0]
        #print "labels.shape=", labels.shape

        indexs = np.array([np.argmax((labels.T)[i]) for i in xrange(batch_size)])
        if typ==CrossEntropy :
            #print z.T
            p = np.vstack([self.softmax((z.T)[i]) for i in xrange(batch_size)]).T
            #print "p.shape=", p.shape
            # l = -np.log(p)
            #print p.T
            #print np.array([p.T[i][indexs[i]] for i in xrange(batch_size)])
            l = -np.log(np.array([p.T[i][indexs[i]] for i in xrange(batch_size)])).T
            #print l
            #print "l.shape=", l.shape
        else :
            pass
        return l, p

    def update(self, p, inputs, labels):
        g = -(labels-p).T
        #print "g.shape=",g.shape
        delta_W = np.dot(g.T,inputs.T) / batch_size + 2 * self.lambda_ * self.W
        delta_b = g / self.batch_size
        #print delta_b
        #print self.b
        #print self.b.shape, delta_b.shape
        self.W += delta_W
        #print "delta_W = ", delta_W
        #print "W after update", self.W
        self.b += delta_b.T

    def ComputeCost(self, loss, batch_size):
        #print "loss=",loss
        cost = np.mean(loss) + self.lambda_ * np.sum(np.square(self.W))
        return cost

    def ComputeAccuracy(self):
        size = 10000
        inputs, labels = self.reader.next_test_data()

        inputs = np.array(inputs)
        #print inputs.shape
        labels = np.array(labels)
        inputs = np.reshape(inputs,(size,3072))
        n = 0.0
        for i in xrange(size//self.batch_size):

        #print inputs.shape
            #print "Acc of batch %d" % i
            z = np.dot(self.W,inputs[self.batch_size*i:self.batch_size*(i+1)].T) + self.b
            s = z.T
            indexs = np.array([np.argmax(labels[i]) for i in xrange(self.batch_size)])
            outputs = np.array([np.argmax(s[i]) for i in xrange(self.batch_size)])
            
            #print indexs, outputs
            
            for i in xrange(batch_size):
                if indexs[i] != outputs[i]:
                    n += 1.0
            #print "n=%d" % n
        acc = 1 - n / size
        
        return acc


    def train(self, loss_type = CrossEntropy, input=None, output=None):
        ep = 0
        #print self.W
#        return
        for ep in xrange(self.max_epoch):
            print("Epoch %d, " % ep)
            error = []
            batch_index = 0
            for batch_index in xrange(50000//self.batch_size):
                #print "W=in batch %d" % batch_index ,self.W
                
                inputs, labels = self.reader.next_train_data(self.batch_size)
                
                # print type(inputs), type(labels)
                inputs = np.array(inputs)
                inputs = inputs / 256
                #print inputs.shape
                labels = np.array(labels)
                
                inputs = np.reshape(inputs,(batch_size,3072)).T
                labels = labels.T
                #print "inputs.shape=",inputs.shape, "labels.shape=",labels.shape
                

                a, p = self.loss(inputs, labels, self.batch_size, loss_type)
                self.update(p, inputs, labels)
                cost = self.ComputeCost(a, batch_size)
                #print("Cost = %f in batch %d\n" % (cost,batch_index))
                #if batch_index==1:
                   # return
            acc = self.ComputeAccuracy()
            print("Accuracy = %f%% after epoch %d" % (acc*100,ep))
            #return




        
  
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
    dr=Cifar10DataReader(cifar_folder="cifar-10-batches-py/")  
    import matplotlib.pyplot as plt  
    d,l=dr.next_test_data()  
    # print np.shape(d),np.shape(l)
    # print l[0]
    # print np.argmax(l[0])
    # print type(d), type(l)
    # plt.imshow(d[0])  
    # plt.show()  
    c = Classifier(32*32*3,10,eta,lamda,epoch,reader=dr)
    # training and testing
    c.train()



    # for i in xrange(60000/batch_size):  
    #     d,l=dr.next_train_data(batch_size=batch_size)  
    #     print np.shape(d),np.shape(l)  
