''' network architecture for Sakuya '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# PYTHONPATH=/data/VideoZoom/codes/models/modules
from DCNv2.dcn_v2 import DCN_sep

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int) Height and width of input tensor as (height, width).
        input_dim: int Number of channels of input tensor.
        hidden_dim: int Number of channels of hidden state.
        kernel_size: (int, int) Size of the convolutional kernel.
        bias: bool Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        # pdb.set_trace()
        # (Pdb) a
        # self = ConvLSTMCell()
        # input_size = (48, 48)
        # input_dim = 64
        # hidden_dim = 64
        # kernel_size = (3, 3)
        # bias = True

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        # pdb.set_trace()
        # self = ConvLSTMCell(
        #   (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # )
        # input_size = (48, 48)
        # input_dim = 64
        # hidden_dim = 64
        # kernel_size = (3, 3)
        # bias = True

    def forward(self, input_tensor, cur_state):
        # pdb.set_trace()
        # torch.Size([1, 64, 272, 480])
        # Pdb) type(cur_state), len(cur_state), cur_state[0].size(), cur_state[1].size()
        # (<class 'list'>, 2, torch.Size([1, 64, 272, 480]), torch.Size([1, 64, 272, 480]))

        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        del i, f, o, g, cc_i, cc_f, cc_o, cc_g, combined, combined_conv, h_cur, c_cur
        torch.cuda.empty_cache()

        # pdb.set_trace()
        # hidden state ht and cell state ct
        # (Pdb) h_next.size(), c_next.size()
        # (torch.Size([1, 64, 272, 480]), torch.Size([1, 64, 272, 480]))

        return h_next, c_next

    def init_hidden(self, batch_size, tensor_size, iscuda=True):
        height, width = tensor_size
        if (iscuda):
            return (torch.zeros(batch_size, self.hidden_dim, height, width).cuda(),
                    torch.zeros(batch_size, self.hidden_dim, height, width).cuda())
        else:
            return (torch.zeros(batch_size, self.hidden_dim, height, width),
                    torch.zeros(batch_size, self.hidden_dim, height, width))

class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        # pdb.set_trace()
        # (Pdb) a
        # self = DeformableConvLSTM(
        #   (cell_list): ModuleList(
        #     (0): ConvLSTMCell(
        #       (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #   )
        # )
        # input_size = (48, 48)
        # input_dim = 64
        # hidden_dim = [64]
        # kernel_size = [(3, 3)]
        # num_layers = 1
        # batch_first = True
        # bias = True
        # return_all_layers = False

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3), input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(
                0), tensor_size=tensor_size, iscuda=input_tensor.is_cuda)

        pdb.set_trace()

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        pdb.set_trace()

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size, iscuda=True):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(
                batch_size, tensor_size, iscuda))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        # pdb.set_trace()
        # (Pdb) type(kernel_size) <class 'tuple'>
        # (Pdb) kernel_size (3, 3)
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        # pdb.set_trace()
        # (Pdb) a
        # param = [(3, 3)]
        # num_layers = 1
        return param

# xxxx 3333 forward


class ConvBLSTM(nn.Module):
    # Constructor
    def __init__(self, input_size, input_dim, hidden_dim,
                 kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):

        super(ConvBLSTM, self).__init__()
        self.forward_net = ConvLSTM(input_size, input_dim, hidden_dims//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias,
                                    return_all_layers=return_all_layers)
        self.reverse_net = ConvLSTM(input_size, input_dim, hidden_dims//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias,
                                    return_all_layers=return_all_layers)

    def forward(self, xforward, xreverse):
        """
        xforward, xreverse = B T C H W tensors.
        """
        pdb.set_trace()
        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)

        if not self.return_all_layers:
            # outputs of last CLSTM layer = B, T, C, H, W
            y_out_fwd = y_out_fwd[-1]
            # outputs of last CLSTM layer = B, T, C, H, W
            y_out_rev = y_out_rev[-1]

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        # reverse temporal outputs.
        y_out_rev = y_out_rev[:, reversed_idx, ...]
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)

        pdb.set_trace()

        return ycat


# PCD -- Pyramid, Cascading and Deformable
class PCD_Align(nn.Module):
    ''' Alignment using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L2_fea_conv_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L1_fea_conv_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # pdb.set_trace()
        # nf = 64
        # groups = 8

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        '''
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(
            torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(
            torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(
            torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
        # pdb.set_trace()
        # (Pdb) type(y), len(y), y[0].size()
        # (<class 'list'>, 1, torch.Size([1, 64, 272, 480]))

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(
            torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(
            torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(
            torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        y = torch.cat(y, dim=1)

        del L1_fea, L2_fea, L3_fea, L1_offset, L2_offset, L3_offset
        torch.cuda.empty_cache()
        # pdb.set_trace()
        # (Pdb) y.size()
        # torch.Size([1, 128, 272, 480])

        return y


class Easy_PCD(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # feature size: f1 = f2 = [B, N, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea1 = [L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(),
                L3_fea[:, 0, :, :, :].clone()]
        fea2 = [L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :, :].clone(),
                L3_fea[:, 1, :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]

        del L1_fea, L2_fea, L3_fea, fea1, fea2
        torch.cuda.empty_cache()

        return fusion_fea


class DeformableConvLSTM(ConvLSTM):
    # xxxx --->
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        ConvLSTM.__init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                          batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        # extract features (for each frame)
        nf = input_dim

        self.pcd_h = Easy_PCD(nf=nf, groups=groups)
        self.pcd_c = Easy_PCD(nf=nf, groups=groups)

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # pdb.set_trace()
        # input_size = (48, 48)
        # input_dim = 64
        # hidden_dim = [64]
        # kernel_size = (3, 3)
        # num_layers = 1
        # front_RBs = 5
        # groups = 8
        # batch_first = True
        # bias = True
        # return_all_layers = False

        # self = DeformableConvLSTM(
        #   (cell_list): ModuleList(
        #     (0): ConvLSTMCell(
        #       (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #   )
        #   (pcd_h): Easy_PCD(
        #     (fea_L2_conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (fea_L2_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (fea_L3_conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (fea_L3_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (pcd_align): PCD_Align(
        #       (L3_offset_conv1_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_offset_conv2_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_dcnpack_1): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_offset_conv1_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv2_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv3_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_dcnpack_1): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_fea_conv_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv1_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv2_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv3_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_dcnpack_1): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L1_fea_conv_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_offset_conv1_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_offset_conv2_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_dcnpack_2): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_offset_conv1_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv2_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv3_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_dcnpack_2): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_fea_conv_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv1_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv2_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv3_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_dcnpack_2): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L1_fea_conv_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
        #     )
        #     (fusion): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        #     (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
        #   )
        #   (pcd_c): Easy_PCD(
        #     (fea_L2_conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (fea_L2_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (fea_L3_conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (fea_L3_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (pcd_align): PCD_Align(
        #       (L3_offset_conv1_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_offset_conv2_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_dcnpack_1): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_offset_conv1_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv2_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv3_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_dcnpack_1): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_fea_conv_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv1_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv2_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv3_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_dcnpack_1): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L1_fea_conv_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_offset_conv1_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_offset_conv2_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L3_dcnpack_2): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_offset_conv1_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv2_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_offset_conv3_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L2_dcnpack_2): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L2_fea_conv_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv1_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv2_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_offset_conv3_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (L1_dcnpack_2): DCN_sep(
        #         (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (L1_fea_conv_2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
        #     )
        #     (fusion): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        #     (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
        #   )
        #   (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
        # )

    def forward(self, input_tensor, hidden_state=None):
        '''        
        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: 
            None. 

        Returns
        -------
        last_state_list, layer_output
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3), input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(
                0), tensor_size=tensor_size, iscuda=input_tensor.is_cuda)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                in_tensor = cur_layer_input[:, t, :, :, :]
                h_temp = self.pcd_h(in_tensor, h)
                c_temp = self.pcd_c(in_tensor, c)
                h, c = self.cell_list[layer_idx](input_tensor=in_tensor,
                                                 cur_state=[h_temp, c_temp])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size, iscuda=True):
        return super()._init_hidden(batch_size, tensor_size, iscuda)


class BiDeformableConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        super(BiDeformableConvLSTM, self).__init__()
        self.forward_net = DeformableConvLSTM(input_size=input_size, input_dim=input_dim, hidden_dim=hidden_dim,
                                              kernel_size=kernel_size, num_layers=num_layers, front_RBs=front_RBs,
                                              groups=groups, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        self.conv_1x1 = nn.Conv2d(2*input_dim, input_dim, 1, 1, bias=True)
        # pdb.set_trace()

    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([1, 3, 64, 272, 480])
        # (Pdb) for i in reversed(range(3)): print(i)
        # 2 1 0
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, reversed_idx, ...]
        out_fwd, _ = self.forward_net(x)
        out_rev, _ = self.forward_net(x_rev)
        # (Pdb) type(out_fwd), len(out_fwd), out_fwd[0].size()
        # (<class 'list'>, 1, torch.Size([1, 3, 64, 272, 480]))
        # (Pdb) type(out_rev), len(out_rev), out_rev[0].size()
        # (<class 'list'>, 1, torch.Size([1, 3, 64, 272, 480]))
        rev_rev = out_rev[0][:, reversed_idx, ...]
        B, N, C, H, W = out_fwd[0].size()
        result = torch.cat((out_fwd[0], rev_rev), dim=2)
        result = result.view(B*N, -1, H, W)
        result = self.conv_1x1(result)
        # pdb.set_trace()
        # (Pdb) result.size()
        # torch.Size([3, 64, 272, 480])
        # (Pdb) B, N, C, H, W
        # (1, 3, 64, 272, 480) ==> (1, 64, 3, 272, 480)
        del out_fwd, out_rev, rev_rev
        torch.cuda.empty_cache()

        return result.view(B, -1, C, H, W)


class VideoZoomModel(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10):
        super(VideoZoomModel, self).__init__()
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        p_size = 48  # a place holder, not so useful
        patch_size = (p_size, p_size)
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.ConvBLSTM = BiDeformableConvLSTM(input_size=patch_size, input_dim=nf, hidden_dim=hidden_dim,
                                              kernel_size=(3, 3), num_layers=1, batch_first=True, front_RBs=front_RBs, groups=groups)
        # reconstruction
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # pdb.set_trace()
        # nf = 64
        # nframes = 3
        # groups = 8
        # front_RBs = 5
        # back_RBs = 40

    def forward(self, x):
        B, N, C, H, W = x.size()  # N input video frames
        # pdb.set_trace() BxTxCxHxW
        # torch.Size([1, 2, 3, 272, 480])
        # ==> torch.Size([1, 3, 3, 1088, 1920])

        # extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        # align using pcd
        to_lstm_fea = []
        '''
        0: + fea1, fusion_fea, fea2
        1: + ...    ...        ...  fusion_fea, fea2
        2: + ...    ...        ...    ...       ...   fusion_fea, fea2
        '''
        # pdb.set_trace()
        # (Pdb) L1_fea.size(), L2_fea.size(), L3_fea.size()
        # (torch.Size([1, 2, 64, 272, 480]),
        # torch.Size([1, 2, 64, 136, 240]),
        # torch.Size([1, 2, 64, 68, 120]))
        # (Pdb) pp N 2

        for idx in range(N-1):
            fea1 = [
                L1_fea[:, idx, :, :, :].clone(), L2_fea[:, idx, :, :,
                                                        :].clone(), L3_fea[:, idx, :, :, :].clone()
            ]
            fea2 = [
                L1_fea[:, idx+1, :, :, :].clone(), L2_fea[:, idx+1, :, :,
                                                          :].clone(), L3_fea[:, idx+1, :, :, :].clone()
            ]
            aligned_fea = self.pcd_align(fea1, fea2)

            fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]
            if idx == 0:
                to_lstm_fea.append(fea1[0])
            to_lstm_fea.append(fusion_fea)
            to_lstm_fea.append(fea2[0])
        lstm_feats = torch.stack(to_lstm_fea, dim=1)

        # pdb.set_trace()
        # (Pdb) lstm_feats.size()
        # torch.Size([1, 3, 64, 272, 480])

        # align using bidirectional deformable conv-lstm
        feats = self.ConvBLSTM(lstm_feats)

        del to_lstm_fea, lstm_feats, L1_fea, L2_fea, L3_fea, \
            fusion_fea, aligned_fea, fea1, fea2
        torch.cuda.empty_cache()

        B, T, C, H, W = feats.size()
        feats = feats.view(B*T, C, H, W)
        out = self.recon_trunk(feats)
        del feats
        torch.cuda.empty_cache()

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        # pdb.set_trace()
        # (Pdb) pp out.size()
        # torch.Size([3, 64, 1080, 1920])

        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        # pdb.set_trace()
        # torch.Size([3, 64, 2160, 3840])

        out = self.lrelu(self.HRconv(out))
        # pdb.set_trace()
        # torch.Size([3, 64, 2160, 3840])

        out = self.conv_last(out)
        # pdb.set_trace()
        # torch.Size([3, 3, 2160, 3840])

        _, _, K, G = out.size()
        outs = out.view(B, T, -1, K, G)
        # pdb.set_trace()
        # (Pdb) out.size()
        # torch.Size([3, 3, 1088, 1920])
        # ==>
        # (Pdb) outs.size()
        # torch.Size([1, 3, 3, 1088, 1920])

        return outs
