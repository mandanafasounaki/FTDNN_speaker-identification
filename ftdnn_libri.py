from  models import SOrthConv, FTDNNLayer, StatsPool, TDNN, DenseReLU, SharedDimScaleDropout
import torch
import torch.nn as nn
import torch.nn.functional as F
from unnorm_load_mfcc import load_data_, prepare_data
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
from sklearn import metrics 
import time

device = torch.device('cuda:0'if torch.cuda.is_available else 'cpu')

class FTDNN(nn.Module):

    def __init__(self, in_dim=24):
        '''
        The FTDNN architecture from
        "State-of-the-art speaker recognition with neural network embeddings in 
        NIST SRE18 and Speakers in the Wild evaluations"
        https://www.sciencedirect.com/science/article/pii/S0885230819302700
        '''
        super(FTDNN, self).__init__()

        self.layer01 = TDNN(input_dim=in_dim, output_dim=512,
                            context_size=5, padding=2)
        self.layer02 = FTDNNLayer(512, 1024, 256, context_size=2, dilations=[
                                  2, 2, 2], paddings=[1, 1, 1])
        self.layer03 = FTDNNLayer(1024, 1024, 256, context_size=1, dilations=[
                                  1, 1, 1], paddings=[0, 0, 0])
        self.layer04 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[
                                  3, 3, 2], paddings=[2, 1, 1])
        self.layer05 = FTDNNLayer(2048, 1024, 256, context_size=1, dilations=[
                                  1, 1, 1], paddings=[0, 0, 0])
        self.layer06 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[
                                  3, 3, 2], paddings=[2, 1, 1])
        self.layer07 = FTDNNLayer(3072, 1024, 256, context_size=2, dilations=[
                                  3, 3, 2], paddings=[2, 1, 1])
        self.layer08 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[
                                  3, 3, 2], paddings=[2, 1, 1])
        self.layer09 = FTDNNLayer(3072, 1024, 256, context_size=1, dilations=[
                                  1, 1, 1], paddings=[0, 0, 0])
        self.layer10 = DenseReLU(1024, 2048)

        self.layer11 = StatsPool()

        self.layer12 = DenseReLU(4096, 512)

        self.layer13 = DenseReLU(512, 512)

        self.lastlayer = nn.Linear(512, 251)

    def forward(self, x):
        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
#        x = nn.BatchNorm1d(in_dim)
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        skip_5 = torch.cat([x_4, x_3], dim=-1)
        x = self.layer05(skip_5)
        x_6 = self.layer06(x)
        skip_7 = torch.cat([x_6, x_4, x_2], dim=-1)
        x = self.layer07(skip_7)
        x_8 = self.layer08(x)
        skip_9 = torch.cat([x_8, x_6, x_4], dim=-1)
        x = self.layer09(skip_9)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)       
        x = self.layer13(x)
        x = self.lastlayer(x)
        x = F.softmax(x,dim=0)
        return x

    def step_ftdnn_layers(self):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.step_semi_orth()

    def set_dropout_alpha(self, alpha):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.dropout.alpha = alpha

    def get_orth_errors(self):
        errors = 0.
        with torch.no_grad():
            for layer in self.children():
                if isinstance(layer, FTDNNLayer):
                    errors += layer.orth_error()
        return errors


def main():
    #Load and prepare data
    train_raw, train_names, train_size = load_data_('/home/ubuntu/projects/voice/speaker_id/libri/dev/train_mfcc')
    train_d, train_l = prepare_data(train_raw, train_names, train_size)
    print("data loaded.")
    

    # Transform the IDs into the format of [0, Number of classes]
    train_ls = []
    for l in train_l:
        train_ls.append(int(l))
    
    train_labels = np.zeros(len(train_ls))
    a = 0
    uniques = np.unique(train_ls)
    for u in uniques:
        ids = [b for b in np.where(train_ls == u)]
        for _id in ids:
            train_labels[_id] = a
        a += 1

    print(np.unique(train_labels))
    
    train_labels = torch.tensor(train_labels).long()
    train_data = torch.tensor(train_d).type('torch.FloatTensor')    

    #Partition the data into train and validation sets
    x_train, x_test, y_train, y_test = train_test_split(train_data, 
        train_labels, 
        test_size = 0.3, 
        random_state=4)
    print("x_train shape: " + str(x_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("x_test shape: " + str(x_test.shape))
    print("y_test shape: " + str(y_test.shape))

    #shuffle training data
    shuffled_idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[shuffled_idx], y_train[shuffled_idx]
    print('Shuffled Data.')

    #Define FTDNN model 
    ftdnn = FTDNN()
    ftdnn.cuda()
    #print(ftdnn)
    
    #Define number of epochs and batch size
    n_epoch = 50
    batch_size = 64
    print('batch size = %d' % (batch_size))
    print('#epochs = %d' % (n_epoch))

    #Loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ftdnn.parameters(), lr = 0.001, momentum =0.9)
   
   #Training the model
    for ep in range(n_epoch):
        start = time.time()
        ftdnn.train()
        running_loss = 0
        c = 0
        for i in range(0, len(x_train), batch_size):
            optimizer.zero_grad()
            data = x_train[i : i+batch_size].cuda()
            #print(data.type())
            labels = y_train[i : i+batch_size].cuda()
            #print("labels shape: " + str(labels.shape))
            outp = ftdnn(data)
            #print("pred shape: " + str(outp.shape))
            loss = criterion(outp, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            c += 1 
           
            #if c % 200 == 199 :
            #    print('[epoch: {0}, step: {1}]------------------------------------ Training Loss = {2}'.format(ep + 1, c+1, running_loss/200))
            #    running_loss == 0
               
        correct = 0 
        total = 0
        train_correct = 0
        train_total = 0
        outputs = []
        train_outputs = []

        #Evaluate the model on training and validation data, save the model with highest accuracy on validation set
        ftdnn.eval()
        with torch.no_grad():
            best_acc = 0
            for x_d in range(0, len(x_train), batch_size):
                train_outputs = ftdnn(x_train[x_d :x_d+batch_size].cuda())
                _, predicted_train = torch.max(train_outputs.data, 1)
                train_total += y_train[x_d:x_d+batch_size].size(0)
                train_correct += (predicted_train.cuda() == y_train[x_d:x_d+batch_size].cuda()).sum().item()
            for x_t in range(0, len(x_test), batch_size):
                outputs = ftdnn(x_test[x_t : x_t+batch_size].cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += y_test[x_t: x_t+batch_size].size(0)
                correct += (predicted.cuda() == y_test[x_t : x_t+batch_size].cuda()).sum().item()
        train_acc = 100 * (train_correct/train_total)
        test_acc = 100 * (correct/total)
        #Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(ftdnn.state_dict(), '/home/ubuntu/projects/voice/speaker_id/libri/dev/FTDNN/checkpoints/24feb/1sec_batchnorm_1skip_001_cc_batch64_' + str(ep) + '.pt')
            end = time.time()
            dur = end - start # duration of training for each epoch
        print("Time : {4} ------- Epoch: {0}--------> Loss = {3} ----------- Training accuracy = {1} ------------ Validation Accuracy = {2}".format(ep+1, train_acc, test_acc, running_loss/c, dur))
#        metrics.plot_det_curve(ftdnn, x_test, y_test)
#        plt.savefig('/okyanus/users/gince/phase2/FTDNN/plots/det_ftdnn_vox.png')

       
            
    print("Finished training.")
            
if __name__ == "__main__":

    main()


