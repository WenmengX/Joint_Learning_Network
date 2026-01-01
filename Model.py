import torch
import torch.nn as nn
import torch.nn.functional as F

import Module as at_module


class FNblock(nn.Module):
    """
    """

    def __init__(self, input_size, hidden_size=254, dropout=0.2, is_online=False, is_first=False):
        """the block of full-band and narrow-band fusion
        """
        super(FNblock, self).__init__()
        self.input_size = input_size
        self.full_hidden_size = hidden_size // 2
        self.is_first = is_first
        self.is_online = is_online
        if self.is_online:
            self.narr_hidden_size = hidden_size
        else:
            self.narr_hidden_size = hidden_size // 2
        self.dropout = dropout

        self.dropout_full = nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True,
                                bidirectional=True)
        if self.is_first:
            self.narrLstm = nn.LSTM(input_size=2 * self.full_hidden_size + self.input_size,
                                    hidden_size=self.narr_hidden_size, batch_first=True,
                                    bidirectional=not self.is_online)
        else:
            self.narrLstm = nn.LSTM(input_size=2 * self.full_hidden_size, hidden_size=self.narr_hidden_size,
                                    batch_first=True, bidirectional=not self.is_online)

    def forward(self, x, nb_skip=None, fb_skip=None):
        # shape of x: nb,nt,nf,nc
        nb, nt, nf, nc = x.shape
        nb_skip = x.permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)
        x = x.reshape(nb * nt, nf, -1)
        if not self.is_first:
            x = x + fb_skip
        x, _ = self.fullLstm(x)
        fb_skip = x
        x = self.dropout_full(x)
        x = x.view(nb, nt, nf, -1).permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)
        if self.is_first:
            x = torch.cat((x, nb_skip), dim=-1)
        else:
            x = x + nb_skip
        x, _ = self.narrLstm(x)
        nb_skip = x
        x = self.dropout_narr(x)
        x = x.view(nb, nf, nt, -1).permute(0, 2, 1, 3)
        return x, fb_skip, nb_skip


class FN_SSL(nn.Module):
    """
    """

    def __init__(self, input_size=4, hidden_size=254, Wdim = 1024, is_online=True, is_doa=False):
        """the block of full-band and narrow-band fusion
        """
        super(FN_SSL, self).__init__()
        self.is_online = is_online
        self.is_doa = is_doa
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wdim = Wdim

        self.block_se1 = FNblock(input_size=1, hidden_size=self.hidden_size, is_online=self.is_online, is_first=True)
        self.block_se12 = FNblock(input_size=self.hidden_size, hidden_size=self.hidden_size, is_online=self.is_online, is_first=False)
        self.se_block2atten = nn.Linear(self.hidden_size, 1)
        self.se_inter = nn.Linear(self.input_size*2, self.input_size)
        self.LNse1 = nn.LayerNorm(self.hidden_size)
        self.LNse2 = nn.LayerNorm(4*300)
        self.se_atten2block = nn.Linear(1, self.hidden_size)
        self.LN2 = nn.LayerNorm(257)
        self.emb2ch1 = nn.Linear(self.hidden_size, 1)

        self.block_1 = FNblock(input_size=self.input_size, is_online=self.is_online, is_first=True)
        self.doa_block2atten = nn.Linear(self.hidden_size, self.input_size)
        self.doa_block2se = nn.Linear(self.hidden_size, self.input_size)
        self.doa_inter = nn.Linear(self.input_size*2, self.input_size)
        self.LNdoa1 = nn.LayerNorm(self.hidden_size)
        self.LNdoa2 = nn.LayerNorm(4 * 300)
        self.doa_atten2block = nn.Linear(self.input_size, self.hidden_size)
        self.block_2 = FNblock(input_size=self.hidden_size, is_online=self.is_online, is_first=False)
        self.block_3 = FNblock(input_size=self.hidden_size, is_online=self.is_online, is_first=False)
        self.emb2ipd = nn.Linear(self.input_size, 2)
        self.pooling = nn.AvgPool2d(kernel_size=(13, 1))
        self.tanh = nn.Tanh()
        if self.is_doa:
            self.ipd2doa = nn.Linear(514, 180)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        nb, nt, nf, nc = x.shape # (1,300,257,4)
        # DOA BLOCK1 ######################################################
        x_doa, fb_skipDOA, nb_skipDOA = self.block_1(x)
        x_doa1 = self.tanh(self.doa_block2atten(x_doa))   # (1,300,257,4)

        # SE BLOCK 1 #######################################################
        x_se = x.permute(0, 2, 1, 3).reshape(nb, nf, nc*nt, 1).permute(0, 2, 1, 3)    # (1,1200,257,1)
        x_se, fb_skip_se, nb_skip_se = self.block_se1(x_se)    # (1,1200,257,254)
        x_se1 = self.tanh(self.se_block2atten(x_se))  # (1,1200,257,1)

        # DOA BLOCK 2 ######################################################
        x_doa2 = torch.cat((x_se1.permute(0,2,1,3).reshape(nb, nf, nt, nc), x_doa1.permute(0,2,1,3)),dim = 3)  # (1,257,300,8)
        x_doa2 = self.tanh(self.doa_inter(x_doa2))    # (1,257,300,4)
        x_doa2 = self.LNdoa2(x_doa2.reshape(nb, nf, nt * nc) + x_doa1.permute(0, 2, 1, 3).reshape(nb, nf, nt*nc)).reshape(nb, nf, nt, nc)  # (1,257,300,4)
        x_doa2 = self.tanh(self.doa_atten2block(x_doa2))  # (1, 257,300,254)
        x_doa2 = self.LNdoa1(x_doa + x_doa2.permute(0, 2,1,3))  # (1, 300,257,254)

        x_doa2, fb_skip, nb_skip = self.block_2(x_doa2, fb_skip=fb_skipDOA, nb_skip=nb_skipDOA)   # (1,300,257,254)
        x_doa2 = self.tanh(self.doa_block2se(x_doa2))  # (1,300,257,4)
        # xJoint, fb_skip, nb_skip = self.block_3(xJoint, fb_skip=fb_skip, nb_skip=nb_skip)
        ####################################################################
        # SE BLOCK 2 #######################################################
        x_se2 = x_se1.permute(0, 2, 1, 3).reshape(nb, nf, nt, nc)  # (1,257,300,4)
        x_se2 = torch.cat((x_se2, x_doa2.permute(0, 2, 1, 3)), dim=3)  # (1,257,300,8)
        x_se2 = self.tanh(self.se_inter(x_se2))

        x_se2 = self.LNse2(
            x_se2.reshape(nb, nf, nt * nc, 1).permute(0, 1, 3, 2) + x_se1.permute(0, 2, 3, 1))  # (1,257,1,1200)
        x_se2 = self.tanh(self.se_atten2block(x_se2.permute(0, 1, 3, 2)))  # (1, 257,1200,254)
        x_se = self.LNse1(x_se + x_se2.permute(0, 2, 1, 3))  # (1, 1200,257,254)

        x_se, fb_skip, nb_skip = self.block_se12(x_se, fb_skip=fb_skip_se, nb_skip=nb_skip_se)  # (1, 1200, 257, 254)
        x_se = self.emb2ch1(x_se)  # (1,1200,257,1)
        x_se = x_se.permute(0, 2, 1, 3).reshape(nb, nf, nt, nc).permute(0, 3, 1, 2)  # (1,4,257,300)

        #########################################################################
        xJoint = x_doa2.permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)
        padded_xJoint = F.pad(xJoint, (0, 0, 0, 12))     # (257,312,4)
        ipd = self.pooling(padded_xJoint)     # (257,24,4)
        ipd = self.tanh(self.emb2ipd(ipd))     # (257,24,2)
        _, nt2, _ = ipd.shape
        ipd = ipd.view(nb, nf, nt2, -1)
        ipd = ipd.permute(0, 2, 1, 3)      # (1,257,24,2)
        ipd_real = ipd[:, :, :, 0]
        ipd_image = ipd[:, :, :, 1]
        result = torch.cat((ipd_real, ipd_image), dim=2)
        if self.is_doa:
            result = self.ipd2doa(result)
        return result, x_se
        #return result


class FN_lightning(nn.Module):
    def __init__(self):
        """the block of full-band and narrow-band fusion
        """
        super(FN_lightning, self).__init__()
        self.arch = FN_SSL()

    def forward(self, x):
        return self.arch(x)


if __name__ == "__main__":
    import torch

    input = torch.randn((2, 4, 256, 298)).cuda()
    net = FN_SSL().cuda()
    ouput = net(input)
    print(ouput.shape)
    print('# parameters:', sum(param.numel() for param in net.parameters()))
