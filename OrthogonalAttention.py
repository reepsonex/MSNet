import  torch
import torch.nn as nn
import torch.functional as ff
import  math
import torch.nn.functional as F

class OrthAtt(nn.Module):
    def __init__(self,ic):
        super(OrthAtt,self).__init__()
        self.pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_y = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_z = nn.AdaptiveAvgPool2d((None, 1))
        self.detal = nn.Parameter(torch.tensor([0.1]))
        self.convdown = nn.Conv2d(in_channels=ic,out_channels=ic //16, kernel_size=1)
        self.convup = nn.Conv2d(in_channels=ic // 16, out_channels=ic, kernel_size=1)

    def forward(self, x):
        input = self.convdown(x)
        B, C, H ,W = input.size()

        #x_attention
        input_x = self.pool_x(input).permute(0, 3, 1, 2).view(B,1,H*C) #[B, 1, H*C]
        T_input_x = input_x.permute(0, 2, 1) #[B, H*C, 1]
        G_x = F.softmax(torch.bmm(T_input_x, input_x),dim=1) #[B, H*C, H*C]
        x_out = torch.bmm(input_x, G_x).permute(0, 2, 1).view(B,C,H,1) #[B, C, H, 1]

        # z_attention
        input_z = self.pool_z(input.permute(0, 1, 3, 2)).permute(0, 3, 1, 2).view(B, 1, -1) # [B, 1, C*W]
        T_input_z = input_z.permute(0, 2, 1)  # [0, C*W, 1]
        G_z = F.softmax(torch.bmm(T_input_z, input_z), dim=1)
        z_out = torch.bmm(input_z, G_z).view(B, C, 1, H)

        #y_attention
        input_y = self.pool_y(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).view(B,1,-1) #[B, 1, H*W]
        T_input_y = input_y.permute(0, 2, 1) #[B, H*W, 1]
        G_y = F.softmax(torch.bmm(T_input_y, input_y),dim=1) #[B, H*W, H*W]
        y_out = torch.bmm(input_y,G_y).view(B, 1, H, W)

        xz_att = self.convup((x_out + z_out)*input)

        out = x*y_out + self.detal*xz_att

        return out





