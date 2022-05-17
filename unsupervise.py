import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import Adam

def norm(x, dim):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    normed = x / torch.sqrt(squared_norm)
    return normed


def singer_spatial_optimize(maps, channel):
    size = maps.size()[3]
    maps_split = torch.split(maps, channel, dim=1)
    for i in range(len(maps_split)):
        singermap = maps_split[i]
        with torch.no_grad():
            spatial_x = singermap.view(maps.size()[0],-1)
            spatial_x = norm(spatial_x, dim=0)
            spatial_x_t = spatial_x.transpose(1, 0)
            G = spatial_x_t @ spatial_x - 1
            G = G.detach().cpu()

        with torch.enable_grad():
            spatial_s = nn.Parameter(torch.sqrt(size**2 * torch.ones((size**2, 1))) / size**2, requires_grad=True)
            spatial_s_t = spatial_s.transpose(1, 0)
            spatial_s_optimizer = Adam([spatial_s], 0.01)

            for iter in range(1):
                f_spa_loss = -1 * torch.sum(spatial_s_t @ G @ spatial_s)
                spatial_s_d = torch.sqrt(torch.sum(spatial_s ** 2))
                if spatial_s_d >= 1:
                    d_loss = -1 * torch.log(2 - spatial_s_d)
                else:
                    d_loss = -1 * torch.log(spatial_s_d)

                all_loss = 50 * d_loss + f_spa_loss

                spatial_s_optimizer.zero_grad()
                all_loss.backward()
                spatial_s_optimizer.step()

        result_map = spatial_s.data.view(1, 1, size, size)

        if i == 0:
            catmap = result_map
        else:
            catmap = torch.cat(([catmap, result_map]),dim=0)

    return catmap


class CCA(nn.Module):
    def __init__(self, ic):
        super(CCA,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.PReLU()
        )
        self.detal = nn.Parameter(torch.tensor([0.2]))

    def forward(self, input):

        map = self.conv(input)
        mask = singer_spatial_optimize(map,1)

        value = mask*input
        out = input + self.detal*value

        return out





