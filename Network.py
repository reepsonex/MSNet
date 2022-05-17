from MLB import *
from unsupervise import *
from  OAB import *
from VGG16 import *
from Conv_block import *


class net(nn.Module):
    def __init__(self, piexl_num, ):
        super(net, self).__init__()
        # (
        #     self.encoder1,
        #     self.encoder2,
        #     self.encoder4,
        #     self.encoder8,
        #     self.encoder16,
        # ) = Backbone_VGG16_in3()

        self.encoder1 = DoubleConv(3, 64)
        self.encoder2 = Down(66, 128)
        self.encoder4 = Down(130, 256)
        self.encoder8 = Down(258, 512)
        self.encoder16 = Down(514, 512)
        self.att1 = OAB(64)
        self.att2 = OAB(128)
        self.att3 = OAB(256)
        self.att4 = OAB(512)

        self.BE = CCA(512)

        self.MM1 =MLB([[2, 2],[4, 4]], in_channels=64, out_channels=2, kernel_size=3)
        self.MM2 = MLB([[2, 2], [4, 4]], in_channels=128, out_channels=2, kernel_size=3)
        self.MM3 = MLB([[2, 2], [4, 4]], in_channels=256, out_channels=2, kernel_size=3)
        self.MM4 = MLB([[2, 2], [4, 4]], in_channels=512, out_channels=2, kernel_size=3)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.maxpoll4 = nn.MaxPool2d(kernel_size=4)
        self.maxpoll2 = nn.MaxPool2d(kernel_size=2)


        self.channel_512 = nn.Conv2d(512 + 66, 64,1)
        self.channel_256 = nn.Conv2d(512 + 130, 64, 1)
        self.channel_128 = nn.Conv2d(512 + 258, 64, 1)
        self.channel_64 = nn.Conv2d(512 + 514, 64, 1)

        self.conv_fusion = nn.Conv2d(64*3, 64, 1)

        self.update1 = Update(piexl_num, 64)
        self.update2 = Update(piexl_num, 64)
        self.update3 = Update(piexl_num, 64)
        self.update4 = Update(piexl_num, 64)

        self.gradients = list()

        self.classifier = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64*4, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def forward(self, x, iter_num):

        f1 = self.encoder1(x)

        # f1 = self.att1(f1)
        f1_m = self.MM1(f1)
        f1 = torch.cat([f1, f1_m], dim=1)

        f2 = self.encoder2(f1)
        f2 = self.att2(f2)
        f2_m = self.MM2(f2)
        f2 = torch.cat([f2, f2_m], dim=1)

        f3 = self.encoder4(f2)
        f3_a = self.att3(f3)
        f3_m = self.MM3(f3_a)
        f3 = torch.cat([f3_a, f3_m], dim=1)


        f4 = self.encoder8(f3)
        f4_a = self.att4(f4)
        f4_m = self.MM4(f4_a)
        f4 = torch.cat([f4_a, f4_m], dim=1)

        f5 = self.encoder16(f4)
        f5 = self.BE(f5)
        f1 = self.maxpoll4(f1)
        f2 = self.maxpoll2(f2)


        n1 = self.Node_initation(f1, f5, 16)

        n2 = self.Node_initation(f2, f5, 8)
        n3 = self.Node_initation(f3, f5, 4)

        n4 = self.Node_initation(f4, f5, 2)


        for epoch in range(iter_num):
            relation1 = self.conv_fusion(torch.cat([self.relation_generation(n1, n2), self.relation_generation(n1, n3), self.relation_generation(n1, n4)], dim=1))
            relation2 = self.conv_fusion(torch.cat([self.relation_generation(n2, n1), self.relation_generation(n2, n3), self.relation_generation(n2, n4)], dim=1))
            relation3 = self.conv_fusion(torch.cat([self.relation_generation(n1, n3), self.relation_generation(n2, n3), self.relation_generation(n3, n4)], dim=1))
            relation4 = self.conv_fusion(torch.cat([self.relation_generation(n1, n2), self.relation_generation(n1, n3), self.relation_generation(n1, n4)], dim=1))


            node1 = self.update1(n1, relation1)
            node2 = self.update2(n2, relation2)
            node3 = self.update3(n3, relation3)
            node4 = self.update4(n4, relation4)

            n1 = node1.clone()
            n2 = node2.clone()
            n3 = node3.clone()
            n4 = node4.clone()

        x = torch.cat([n1, n2, n3, n4],dim=1)
        # x.register_hook(self.save_gradient)
        out = self.classifier(x)
        # return out, self.gradients, x
        return  out


    def Node_initation(self, f1, f2, scale_factor):

        if scale_factor == 16:
            f2 = self.up4(f2)
            out = self.channel_512(torch.cat([f1, f2],dim=1))
        elif scale_factor == 8:
            f2 = self.up4(f2)
            out = self.channel_256(torch.cat([f1, f2],dim=1))
        elif scale_factor == 4:
            f2 = self.up4(f2)
            out = self.channel_128(torch.cat([f1, f2],dim=1))
        else:
            f2 = self.up4(f2)
            f1 = self.up2(f1)
            out = self.channel_64(torch.cat([f1, f2],dim=1))


        return out

    def relation_generation(self, f1, f2):

        channel_reduction_f1 = nn.AdaptiveAvgPool2d((None, 1))(f1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        channel_reduction_f2 = nn.AdaptiveAvgPool2d((None, 1))(f2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        B, C, H, W = channel_reduction_f2.size()
        f1_w = channel_reduction_f1.view(B, 1, H * W)
        f2_w = channel_reduction_f2.view(B, 1, H * W)
        # f2_y = f2_w.transpose(B, 1, -1)  # [B, 1, H*W]
        f1_w = f1_w.permute(0, 2, 1)  # [B, H*W, 1]
        G_y = F.softmax(torch.bmm(f1_w, f2_w), dim=1)  # [B, H*W, H*W]
        attout = torch.bmm(G_y, f1_w).view(B, 1, H, W)
        weight_mask = nn.Sigmoid()(attout)

        out = f1 * weight_mask
        return  out

class Update(nn.Module):

    def __init__(self, pixel_num, in_channels):
        super(Update, self).__init__()

        self.linear  = nn.Sequential(
            nn.Linear(pixel_num*pixel_num, pixel_num),
            nn.LayerNorm([in_channels,pixel_num]),
            nn.Linear(pixel_num, pixel_num*pixel_num),
            nn.LayerNorm([in_channels, pixel_num*pixel_num]),
            nn.ReLU()

        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
    def forward(self, f_next, f):

        feature_fusion = self.conv(torch.cat([f_next, f],dim=1))
        B, C, H, W = feature_fusion.size()
        feature_trans = feature_fusion.view(B, C, H*W)
        feature_update = self.linear(feature_trans)

        out = feature_update.view(B, C, H, W)


        return out


