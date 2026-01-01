import torch
import torch.nn as nn
import torch.nn.functional as F
from base.norm import *
from base.non_linear import *
from base.linear_group import LinearGroup
from torch import Tensor
from torch.nn import MultiheadAttention
from utils import forgetting_norm
#torch.cuda.set_device(1)
import Module as at_module

class SpatialNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)   # LN
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        self.tconvffn = nn.ModuleList([
            new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)   # (1,257,300,96)
        x = x + self._full(x)    # (1,257,300,96)
        x = x + self._fconv(self.fconv2, x)   # (1,257,300,96)
        x_, attn = self._tsa(x, att_mask)    # (1,257,300,96)  None
        x = x + x_     # (1,257,300,96)
        x = x + self._tconvffn(x)   # (1,257,300,96)
        return x, attn

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)   # (1,257,300,96)
        x = x.reshape(B * F, T, H)   # (257,300,96)
        need_weights = False if hasattr(self, "need_weights") else self.need_weights   # False
        x, attn = self.mhsa.forward(x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=attn_mask)  # (257,300,96)
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]    # (1,257,96,300)
        x = x.reshape(B * F, H0, T)    # (257,96,300)
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)   # (1,257,96,300)
        x = x.transpose(-1, -2)  # [B,F,T,H] # (1,257,300,96)
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F] # (1,300,96,257)
        x = x.reshape(B * T, H, F)   # (300,96,257)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H] (1,257,300,96)
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)     # (1,257,300,96)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]  # (1,300,96,257)
        x = x.reshape(B * T, H, F)   # (300,96,257)
        x = self.squeeze(x)  # [B*T,H',F] (300,8,257)  Linear + SiLU
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]  Linear Groups (300,8,257)
        x = self.unsqueeze(x)  # [B*T,H,F]  Linear + SiLU (300,96,257)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]  # (1,257,300,96)
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"

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
        nb_skip = x.permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)   # (257,300,4) (257,1200,1)
        x = x.reshape(nb * nt, nf, -1)   # (300,257,4)  (1200,257,1)
        if not self.is_first:
            x = x + fb_skip
        x, _ = self.fullLstm(x)   # (300,257,254)   (1200,257,254)
        fb_skip = x    # (300,257,254) (1200,257,254)
        x = self.dropout_full(x)   # (300,257,254) (1200,257,254)
        x = x.view(nb, nt, nf, -1).permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)   # (257,300,254)  (257,1200,254)
        if self.is_first:
            x = torch.cat((x, nb_skip), dim=-1)  # (257,300,258) (257,1200,255)
        else:
            x = x + nb_skip
        x, _ = self.narrLstm(x)   # (257,300,254)  (257,1200,254)
        nb_skip = x   # (257,300,254)  (257,1200,254)
        x = self.dropout_narr(x)    # (257,300,254)  (257,1200,254)
        x = x.view(nb, nf, nt, -1).permute(0, 2, 1, 3)    # (1,300,257,254) (1,1200，257,254)
        return x, fb_skip, nb_skip


class Joint_Learning(nn.Module):
    """
    """

    def __init__(self, input_size=4,
                 hidden_size=254,
                 Wdim = 1024,
                 is_online=True,
                 is_doa=False,
                 dim_input=4,  # the input dim for each time-frequency point
                 dim_output=4,  # the output dim for each time-frequency point
                 dim_squeeze=8,
                 num_layers=8,
                 num_freqs=257,
                 encoder_kernel_size: int = 5,
                 dim_hidden: int = 96,
                 dim_ffn: int = 192,
                 num_heads: int = 2,
                 dropout: Tuple[float, float, float] = (0, 0, 0),
                 kernel_size: Tuple[int, int] = (5, 3),
                 conv_groups: Tuple[int, int] = (8, 8),
                 norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
                 padding: str = 'zeros',
                 full_share: int = 0,  # share from layer 0
                 ):
        """the block of full-band and narrow-band fusion
        """
        super(Joint_Learning, self).__init__()
        self.num_freqs = num_freqs
        self.num_layers = num_layers
        self.dim_squeeze = dim_squeeze
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.is_online = is_online
        self.is_doa = is_doa
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wdim = Wdim
        ########################## SE module ##########################################
        # encoder
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")
        self.encoder2 = nn.Conv1d(in_channels=dim_input, out_channels= self.hidden_size, kernel_size=encoder_kernel_size,  stride=1, padding="same")
        # spatialnet layers
        full = None
        layers = []
        for l in range(num_layers):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)
        #####################################################################################
        self.block_1 = FNblock(input_size=self.input_size, is_online=self.is_online, is_first=True)

        self.block_2 = FNblock(input_size=self.hidden_size, is_online=self.is_online, is_first=False)
        self.block_3 = FNblock(input_size=self.hidden_size, is_online=self.is_online, is_first=False)
        self.emb2ipd = nn.Linear(self.hidden_size, 2)
        self.emb2ipd2 = nn.Linear(self.hidden_size, 2)
        self.hidden2ch = nn.Linear(self.hidden_size,4)
        self.ch2phase = nn.Linear(4,2)
        self.pooling = nn.AvgPool2d(kernel_size=(13, 1))

        self.cnn_deep = 4
        self.cnn_conv = nn.ModuleList(
            [nn.Conv2d(in_channels= self.input_size*2, out_channels=self.hidden_size//4, kernel_size= (5,5), stride= (2,2))])
        self.cnn_conv += nn.ModuleList(
            [nn.Conv2d(in_channels=self.hidden_size//4, out_channels=self.hidden_size//2, kernel_size= (5,5), stride= (2,2))])
        self.cnn_conv += nn.ModuleList(
            [nn.Conv2d(in_channels=self.hidden_size // 2, out_channels=self.hidden_size, kernel_size= (5,5), stride= (2,2))])
        self.cnn_conv += nn.ModuleList([nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(5, 5),stride=(2, 2))])
        self.in_prelu = nn.ModuleList([nn.PReLU(self.hidden_size//4)])
        self.in_prelu += nn.ModuleList([nn.PReLU(self.hidden_size // 2)])
        self.in_prelu += nn.ModuleList([nn.PReLU(self.hidden_size)])
        self.in_prelu += nn.ModuleList([nn.PReLU(self.hidden_size)])

        self.cnnT_deep = 4
        self.convT0 = nn.ConvTranspose2d(in_channels = self.hidden_size, out_channels = self.hidden_size, kernel_size= (5,5), stride= (2,2), output_padding = (0,1))
        self.T_prelu0 = nn.PReLU(self.hidden_size)
        self.convT1 = nn.ConvTranspose2d(in_channels=self.hidden_size*2, out_channels=self.hidden_size //2, kernel_size= (5,5), stride= (2,2), output_padding = (1,1))
        self.T_prelu1 = nn.PReLU(self.hidden_size //2)
        self.convT2 = nn.ConvTranspose2d(in_channels= self.hidden_size, out_channels=self.hidden_size //4, kernel_size= (5,5), stride= (2,2), output_padding = (0,1))  # output_padding = (0,1) for Librispeech   (0,0) for DNS3
        self.T_prelu2 = nn.PReLU(self.hidden_size //4)
        self.convT3 = nn.ConvTranspose2d(in_channels= 63*2, out_channels=self.input_size * 2, kernel_size=(5, 5), stride=(2, 2), output_padding=(0, 1))
        self.T_prelu3 = nn.PReLU(self.input_size * 2)

        self.doa2se = nn.Linear(self.input_size*2, self.input_size)
        self.SeDoaCat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tanh = nn.Tanh()


    def forward(self, data_x, eps = 1e-6, return_attn_score=False):
        r"""
               Args:
                   x: shape [B, 2C, F, T] raw signals in magnitude-phase format
                   x_RIRI: shape [B, F, T, 2C] raw signals in real-imaginary format

               Outputs:
                   result2: shape [B, T', 2F], T' is the temporal dimension after pooling
                   x_se2: shape [B, F, T, 2C]
        """
        x, x_RIRI = data_x   # (1,4,257,300)(1,257,300,4)
        x = x.permute(0, 3, 2, 1)
        #x = x_RIRI.permute(0,2,1,3)
        nb, nt, nf, nc = x.shape  # (1,300,257,4)
        x_doa = x
        # DOA Block 1 #############################################################
        x_doa, fb_skipDOA, nb_skipDOA = self.block_1(x_doa)    # (1,300,257,254)
        x_doa2, fb_skip, nb_skip = self.block_2(x_doa, fb_skip=fb_skipDOA, nb_skip=nb_skipDOA)  # (1,300,257,254)

        xJoint = x_doa2.permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)  # (257,300,254)
        padded_xJoint = F.pad(xJoint, (0, 0, 0, 12))  # (257,312,254)
        ipd = self.pooling(padded_xJoint)  # (257,24,254)
        del x_doa, fb_skipDOA, nb_skipDOA


        ipd = self.tanh(self.emb2ipd(ipd))  # (257,24,2)
        _, nt2, _ = ipd.shape
        ipd = ipd.view(nb, nf, nt2, -1)  # (1,257,24,2)
        ipd = ipd.permute(0, 2, 1, 3)  # (1,24,257,2)
        ipd_real = ipd[:, :, :, 0]
        ipd_image = ipd[:, :, :, 1]
        result = torch.cat((ipd_real, ipd_image), dim=2)
        del ipd, ipd_real, ipd_image
        # SE ############################################################### ipd0: (257,24,4)
        Doa2Se0 = torch.cat((x_RIRI, self.tanh(self.hidden2ch(xJoint)).reshape(nb, nf, nt, nc)), dim=3).permute(0, 3, 1, 2)
        Doa2Se1 = self.in_prelu[0](self.cnn_conv[0](Doa2Se0))
        Doa2Se2 = self.in_prelu[1](self.cnn_conv[1](Doa2Se1))
        Doa2Se3 = self.in_prelu[2](self.cnn_conv[2](Doa2Se2))
        Doa2Se4 = self.in_prelu[3](self.cnn_conv[3](Doa2Se3))

        Doa2Se3T = self.T_prelu0(self.convT0(Doa2Se4))
        Doa2Se2T = self.T_prelu1(self.convT1(torch.cat((Doa2Se3T, Doa2Se3), dim=1))) + Doa2Se2
        Doa2Se1T = self.T_prelu2(self.convT2(torch.cat((Doa2Se2T, Doa2Se2), dim=1))) + Doa2Se1
        Doa2Se0T = self.T_prelu3(self.convT3(torch.cat((Doa2Se1T, Doa2Se1), dim=1))) + Doa2Se0

        #x_RIRI = self.doa2se(Doa2Se.permute(0,2,3,1)) + x_RIRI
        x_RIRI = self.doa2se(Doa2Se0T.permute(0, 2, 3, 1)) + x_RIRI

        x_se = self.encoder(x_RIRI.reshape(nb * nf, nt, nc).permute(0,2,1)).permute(0, 2, 1)  # (257,300,96)
        H = x_se.shape[2]    # 96
        attns = [] if return_attn_score else None
        x_se = x_se.reshape(nb, nf, nt, H)   # (1,257,300,96)
        for m in self.layers:
            setattr(m, "need_weights", return_attn_score)
            x_se, attn = m(x_se)   # (1,257,300,96)
            if return_attn_score:
                attns.append(attn)
        x_se2 = self.decoder(x_se)   # (1,257,300,4)
        #x_se2 = x_se2.permute(0, 3, 1, 2)    # (1,4,257,300)

        # DOA Block 2 #############################################################   # xJoint (257,300,254)   #x_doa2 (1,300,257,254)
        x_se2_1 = torch.view_as_complex(torch.stack((x_se2[..., 0], x_se2[..., 1]), dim=-1))
        x_se2_2 = torch.view_as_complex(torch.stack((x_se2[..., 2], x_se2[..., 3]), dim=-1))
        x_se_rebatch = torch.stack((x_se2_1, x_se2_2), dim=-1)
        del x_se2_1, x_se2_2
        x_se_rebatch = x_se_rebatch.permute(0, 3, 1, 2)
        x_se_mag = torch.abs(x_se_rebatch)
        x_se_phase = torch.angle(x_se_rebatch)
        mean_value = forgetting_norm(x_se_mag)
        x_se_mag = x_se_mag / (mean_value + eps)

        x_se2doa = torch.cat((x_se_mag, x_se_phase), dim=1).permute(0, 2,3,1)    # (1,257,300,4)
        del x_se_rebatch, x_se_mag, x_se_phase, mean_value

        x_se3 = self.encoder2(x_se2doa.reshape(nb*nf, nt, nc).permute(0,2,1)).permute(0,2,1)   # (257, 300, 254)
        Se2DoaJoint = self.tanh(self.SeDoaCat(torch.cat((x_se3, x_doa2.permute(0,2,1,3).reshape(nb*nf, nt, self.hidden_size)), dim = 2)))    # (257,300,254)
        #Se2DoaJoint, _ = self.LstmSe2Doa(Se2DoaJoint)
        Se2DoaJoint = Se2DoaJoint.reshape(nb,nf,nt,self.hidden_size).permute(0,2,1,3)

        x_doa2 = Se2DoaJoint + x_doa2
        del x_se3, Se2DoaJoint

        x_doa2, fb_skip, nb_skip = self.block_3(x_doa2, fb_skip=fb_skip, nb_skip=nb_skip)  # (1,300,257,254)

        xJoint = x_doa2.permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)  # (257,300,254)
        padded_xJoint = F.pad(xJoint, (0, 0, 0, 12))  # (257,312,254)
        ipd2 = self.pooling(padded_xJoint)  # (257,24,254)
        del x_doa2

        ipd2 = self.tanh(self.emb2ipd2(ipd2))  # (257,24,2)
        _, nt2, _ = ipd2.shape
        ipd2 = ipd2.view(nb, nf, nt2, -1)  # (1,257,24,2)
        ipd2 = ipd2.permute(0, 2, 1, 3)  # (1,24,257,2)
        ipd2_real = ipd2[:, :, :, 0]
        ipd2_image = ipd2[:, :, :, 1]
        result2 = torch.cat((ipd2_real, ipd2_image), dim=2)
        del ipd2, ipd2_real, ipd2_image
        #########################################################################
        return result2, x_se2

if __name__ == "__main__":
    import torch

    input_MagnitudePhase = torch.randn((1, 4, 257, 300)).cuda()
    input_RealImage = torch.randn((1,257,300,4)).cuda()
    input = (input_MagnitudePhase, input_RealImage)
    net = Joint_Learning().cuda() # (1,4,257,300)(1,257,300,4)
    ouput = net(input)
    # print(ouput.shape)
    print('# parameters:', sum(param.numel() for param in net.parameters()))


