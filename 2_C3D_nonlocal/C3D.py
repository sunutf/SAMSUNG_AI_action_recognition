import torch
import torch.nn as nn
from non_local import NonLocalBlock3D 

class C3D(nn.Module):
    """
    The C3D network.
    """
    
    def __init__(self, num_classes, pretrained=""):
        super(C3D, self).__init__()
        
        #============================================================
        #All of convolution layers use kernel_size (3,3,3) and padding (1, 1, 1)
        #conv1 3 -> 64
        #conv2 64 -> 128
        #conv3a 128 -> 256
        #conv3b 256 -> 256
        #conv4a 256 -> 512
        #conv4b 512 -> 512
        #conv5a 512 -> 512
        #conv5b 512 -> 512
        #fc6 (you need to find input channel size) -> 4096
        #fc7 4096 -> num_classes
        #============================================================

        self.conv1 = 
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        #self.nonlocal1 = NonLocalBlock3D(64)

        self.conv2 = 
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #self.nonlocal2 = NonLocalBlock3D(128)

        self.conv3a = 
        self.conv3b = 
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #self.nonlocal3 = NonLocalBlock3D(256)

        self.conv4a = 
        self.conv4b = 
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #self.nonlocal4 = NonLocalBlock3D(512)

        self.conv5a = 
        self.conv5b = 
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = 
        self.fc7 = 
        #============================================================

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights(pretrained)

    def forward(self, x):
        
        #============================================================
        #use all layer to forward
        #============================================================
  
        
        #============================================================
        logits = self.fc7(x)

        return logits

    def __load_pretrained_weights(self, model_path):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(model_path)['state_dict']
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=11)

    outputs = net.forward(inputs)
    print(outputs.size())
