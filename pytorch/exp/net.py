import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 16)
#        self.relu1=nn.ReLU()
        self.fc2 = nn.Linear(16, 32)
#        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
#        self.relu3=nn.ReLU()



    def forward(self, x):
        x=F.leaky_relu(self.fc1(x))
        x=F.leaky_relu( self.fc2(x))
        x=F.leaky_relu( self.fc3(x))
        #x=self.relu1(self.fc1(x))
        #x=self.relu2( self.fc2(x))
        #x=self.relu3( self.fc3(x))
        return x

    # def get(self,x):
    #     return model(Variable(torch.from_numpy(np.asarray([x],dtype=np.float32)))).data[0]


