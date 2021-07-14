import torch
from torch import nn
from torch.nn import functional as F

class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels, test_mode=False, dimension=3, sub_sample=True):
        super(NonLocalBlock3D, self).__init__()
        
        self.test_mode = test_mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels

        self.inter_channels = in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
       
        #============================================================
        #make self.g , self.theta, self.phi
        #these are nn.Conv3d, 1x1x1, stride=1, padding=0
        #============================================================
        self.g = 
        
        self.theta = 

        self.phi = 
        #============================================================

        #============================================================
        #make self.W
        #in this part, self.W.weight and self.W.bias must initialize to 0
        #============================================================
        self.W =
        nn.init.constant_(self.W.bias, 0)
        #============================================================

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)
        #============================================================
        #1. use self.g(x)
        #2. use self.theta(x)
        #3. use self.phi(x)
        #4. several matrix multiplication between previous return value
        #5. use self.W(y)
        #6. make z with x and self.W(y)
        #============================================================
        g_x = 
        g_x = g_x.permute(0, 2, 1)

        theta_x =
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = 

        f = 
        f_div_C = 
        
        y =         
        if self.test_mode:
            print("x: {}".format(x.shape))
            print("g_x: {}".format(g_x.shape))
            print("theta_x: {}".format(theta_x.shape))
            print("phi_x: {}".format(phi_x.shape))
            print("f: {}".format(f.shape))
            print("y: {}".format(y.shape))

        y = y.permute(0, 2, 1).contiguous()
        y = 
        W_y = 
        z = 
        #============================================================

        return z


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = False

    img = Variable(torch.randn(2, 3, 10, 20, 20))
    net = NonLocalBlock3D(3, test_mode=True, sub_sample=sub_sample)
    out = net(img)
    print(out.size())
