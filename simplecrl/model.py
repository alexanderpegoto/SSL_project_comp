import torch.nn as nn
import torchvision.models as models
import torch




class ResNet(nn.Module):
    def __init__(self,base_model='resnet_50',out_dim=128):
        super(ResNet, self).__init__()

        # Load ResNet
        resnet = models.resnet50(pretrained = False, num_classes = out_dim)
        # Encode everything except last fc layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        dim_mlp = resnet.fc.in_features
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp,out_dim)
        )
        
    def forward(self,x):
        h = self.encoder(x) # [b,input,1,1]
        h = torch.flatten(h,1) # [b,i]
        z = self.projection_head(h)
        return h,z
    

         
        
        