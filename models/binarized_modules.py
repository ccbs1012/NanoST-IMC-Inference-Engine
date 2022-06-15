import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn import functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import  _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union



# from common_types import _size_1_t, _size_2_t, _size_3_t
# from torch.nn.modules.conv.lazy import LazyModuleMixin

import numpy as np


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def Ternarize(tensor):

    return tensor.round()


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output



def Quantize(tensor, numBits=3, min_val = -1, max_val = 1):
    
    # for -1 ~ +1
    scale = (max_val - min_val)/(2**numBits-1)

    tensor = ((tensor - min_val).div(scale)).round().mul(scale) + min_val
    return tensor

def Log_Quantize(tensor, num_state=8):
    tensor_s = torch.log2(torch.abs(tensor))
    tensor_s[tensor_s<(-(num_state/2-1))] = -(num_state/2-1)
    tensor = 2**(tensor_s.round())*tensor.sign()
    return tensor  
    
def Log_Quantize_b(tensor, num_state=8, base = torch.tensor(100**(1/3)).type(torch.cuda.FloatTensor) ):
    tensor_s = torch.log(torch.abs(tensor)).div(torch.log(base))
    tensor_s[tensor_s<(-(num_state/2-1))] = -(num_state/2-1)
    tensor = base**(tensor_s.round())*tensor.sign()
    return tensor
    
    


# class BinarizeLinear_train(nn.Linear):

    # def __init__(self, precision: int=3, *kargs, **kwargs):
        # super(BinarizeLinear_train, self).__init__(precision,*kargs, **kwargs)

        # self.precision = precision

    # def forward(self, input):

        # if input.size(1) != 784:
            # input.data=Binarize(input.data).add(1).mul(.5)
            
        # if not hasattr(self.weight,'org'):
            # self.weight.org=self.weight.data.clone()

        # print(self.precision)
        # self.weight.data=Quantize(self.weight.org, numBits = self.precision)
        # print(self.weight.data[0:2,0:2])
        
        # out = nn.functional.linear(input, self.weight)
        # if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            # out += self.bias.view(1, -1).expand_as(out)

        # return out
        
class BinarizeLinear_train(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, precision_w: int = 3, precision_a: int =1, comp: bool = False) -> None:
        super(BinarizeLinear_train, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.precision_w = precision_w
        self.precision_a = precision_a
        self.comp = comp

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        if input.size(1) != 784:
            if not self.comp:
                input.data=Quantize(input.data, numBits = self.precision_a).add(1).mul(.5)
            else:
                input.data=Quantize(input.data, numBits = self.precision_a)
            
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        self.weight.data=Quantize(self.weight.org, numBits = self.precision_w)

        
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

        

class BinarizeConv2d_train(_ConvNd):

    def __init__(self,in_channels: int,out_channels: int,kernel_size: _size_2_t,stride: _size_2_t = 1,padding: Union[str, _size_2_t] = 0,dilation: _size_2_t = 1,groups: int = 1,bias: bool = False,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        precision_w: int = 3, 
        precision_a: int =1, 
        comp: bool = False
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(BinarizeConv2d_train, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
            
        self.precision_w = precision_w
        self.precision_a = precision_a
        self.comp = comp
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if input.size(1) != 3:
            if not self.comp:
                input.data=Quantize(input.data, numBits = self.precision_a).add(1).mul(.5)
            else:
                input.data=Quantize(input.data, numBits = self.precision_a)
            
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Quantize(self.weight.org, numBits = self.precision_w)
        
        out = nn.functional.conv2d(input, self.weight, bias=None, stride=self.stride,
                                   padding=self.padding, dilation=1, groups=1)

        # if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            # out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out



class BinarizeLinear_inference(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, precision_a: int =1, comp: bool = False) -> None:
        super(BinarizeLinear_inference, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.precision_a = precision_a
        self.comp = comp

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        if input.size(1) != 784:
            if not self.comp:
                input.data=Quantize(input.data, numBits = self.precision_a).add(1).mul(.5)
            else:
                input.data=Quantize(input.data, numBits = self.precision_a)
            

        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


        
class BinarizeConv2d_inference(_ConvNd):

    def __init__(self,in_channels: int,out_channels: int,kernel_size: _size_2_t,stride: _size_2_t = 1,padding: Union[str, _size_2_t] = 0,dilation: _size_2_t = 1,groups: int = 1,bias: bool = False,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        precision_a: int =1, 
        comp: bool = False
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(BinarizeConv2d_inference, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
            
        self.precision_a = precision_a
        self.comp = comp
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if input.size(1) != 3:
            if not self.comp:
                input.data=Quantize(input.data, numBits = self.precision_a).add(1).mul(.5)
            else:
                input.data=Quantize(input.data, numBits = self.precision_a)
            
        
        out = nn.functional.conv2d(input, self.weight, bias=None, stride=self.stride,
                                   padding=self.padding, dilation=1, groups=1)

        # if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            # out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
        
       
class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784: 
           input.data=Binarize(input.data).add(1).mul(.5)
           

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

                
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        # binary activations
        if input.size(1) != 3:
            input.data = Binarize(input.data).add(1).mul(.5)
           
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org) #original BNN
        
        
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class crxb_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, ir_drop, gmax, gmin, gwire,
                stride=1, padding=0, dilation=1,
                 groups=1, bias=False, crxb_size=64, quantize=8):
        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)

        assert self.groups == 1, "currently not support grouped convolution for custom conv"

        self.ir_drop = ir_drop

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size

        self.nchout_index = nn.Parameter(torch.arange(self.out_channels), requires_grad=False).cuda()
        weight_flatten = self.weight.view(self.out_channels, -1)
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            weight_flatten.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad(
            weight_flatten.shape[0], self.crxb_size)
        self.h_out = None
        self.w_out = None
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)

        ################# Hardware conversion ##############################

        # ReRAM cells
        self.Gmax = gmax  # max conductance
        self.Gmin = gmin  # min conductance
        self.Gwire = gwire
        self.scale = torch.tensor(1.)/(gmax - gmin)


    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data).add(1).mul(.5)
           
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org) #original BNN

        # 2. Perform the computation between input voltage and weight conductance
        self.h_out = int((input.size(2) - self.kernel_size[0]  + 2 * self.padding[0]) / self.stride[0] + 1) #output feature height 
        self.w_out = int((input.size(3) - self.kernel_size[0]  + 2 * self.padding[0]) / self.stride[0] + 1) #output feature height 


        # 2.1 flatten and unfold the weight and input
        input_unfold = F.unfold(input, kernel_size=self.kernel_size[0],
                                dilation=self.dilation, padding=self.padding,
                                stride=self.stride)
        weight_flatten = self.weight.reshape(self.out_channels,-1)

        # 2.2. add paddings
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)
        input_padded = F.pad(input_unfold, self.input_pad,
                             mode='constant', value=0)
        # 2.3. reshape to crxb size
        input_crxb = input_padded.reshape(input.size(0), self.crxb_row, self.crxb_size, input_padded.size(2))
        # print(input.size())
        weight_crxb = weight_padded.reshape(self.crxb_col, self.crxb_size, self.crxb_row, self.crxb_size).transpose(1, 2)

        # convert the floating point weight into conductance pair values
        G_pos = torch.zeros(weight_crxb.size()).cuda(); G_pos[weight_crxb==1] = self.Gmax; G_pos[weight_crxb==-1] = self.Gmin
        G_neg = torch.zeros(weight_crxb.size()).cuda(); G_neg[weight_crxb==1] = self.Gmin; G_neg[weight_crxb==-1] = self.Gmax


        # 2.4. compute matrix multiplication followed by reshapes


        # this block is to calculate the ir drop of the crossbar
        if self.ir_drop:
            input_new = input_crxb.repeat(self.crxb_col,self.crxb_size,1,1,1,1).permute(2,0,3,4,1,5) #(batch,col,row,m,n,feature)

            Rl_norm_pos = self.Gwire*G_pos.repeat(input.size(0),input_padded.size(2),1,1,1,1).permute(0,2,3,5,4,1)  #(batch,col,row,m,n,feature)
            Rl_norm_neg = self.Gwire*G_neg.repeat(input.size(0),input_padded.size(2),1,1,1,1).permute(0,2,3,5,4,1)  #(batch,col,row,m,n,feature)
            Rl_norm_pos[input_new==0] = 0; Rl_norm_neg[input_new==0] = 0
            Rl_norm_pos = torch.flip(Rl_norm_pos,[3]); Rl_norm_neg = torch.flip(Rl_norm_neg,[3])
            array_factor_pos = torch.cumsum(Rl_norm_pos,dim=3); array_factor_neg = torch.cumsum(Rl_norm_neg,dim=3)


            Rl_norm_pos[Rl_norm_pos!=0] = Rl_norm_pos[Rl_norm_pos!=0]/(tuning_norm(array_factor_pos[Rl_norm_pos!=0],torch.tensor(array_size)))
            Rl_norm_neg[Rl_norm_neg!=0] = Rl_norm_neg[Rl_norm_neg!=0]/(tuning_norm(array_factor_neg[Rl_norm_neg!=0],torch.tensor(array_size)))

            Rl_norm_pos = torch.cumsum(Rl_norm_pos,dim=3)
            Rl_norm_neg = torch.cumsum(Rl_norm_neg,dim=3)


            Vbl_pos = torch.flip( torch.cumsum(Rl_norm_pos,dim=3),[3]) #(batch,col,row,m,n,feature)
            Vbl_neg = torch.flip( torch.cumsum(Rl_norm_neg,dim=3),[3]) #(batch,col,row,m,n,feature)

            G_pos = G_pos.repeat(input.size(0),input_padded.size(2),1,1,1,1).permute(0,2,3,5,4,1) 
            G_neg = G_neg.repeat(input.size(0),input_padded.size(2),1,1,1,1).permute(0,2,3,5,4,1) #(batch,col,row,m,n,feature)

            # Vwl_pos = (input_new-Vbl_pos)*((1/G_pos)/(Rwl+1/G_pos))
            Vwl_pos = (input_new-Vbl_pos)*((1/self.Gwire)/(1/self.Gwire+G_pos))

            # Vwl_neg = (input_new-Vbl_neg)*((1/G_neg)/(Rwl+1/G_neg))
            Vwl_neg = (input_new-Vbl_neg)*((1/self.Gwire)/(1/self.Gwire+G_neg))

            Vwl_pos[Vwl_pos<0] = 0; Vwl_neg[Vwl_neg<0] = 0

            input_new_pos = Vwl_pos-Vbl_pos; input_new_pos[input_new_pos<0] = 0
            input_new_neg = Vwl_neg-Vbl_neg; input_new_neg[input_new_neg<0] = 0

            output_crxb = torch.sum(input_new_pos*G_pos, dim=3) - torch.sum(input_new_neg*G_neg, dim=3)

        else:
            input_crxb = input_padded.reshape(input.size(0), 1, self.crxb_row, self.crxb_size, input_padded.size(2))
            G_crxb = torch.cat((G_pos.unsqueeze(0), G_neg.unsqueeze(0)), 0)
            output_crxb = torch.matmul(G_crxb[0], input_crxb) - torch.matmul(G_crxb[1], input_crxb)
                          

        # 3. perform ADC operation (i.e., current to digital conversion)
        output_sum = torch.sum(output_crxb, dim=2)

        output = output_sum.reshape(output_sum.size(0),output_sum.size(1) * output_sum.size(2), self.h_out, self.w_out).index_select(dim=1, index=self.nchout_index)
        output = output.mul(self.scale)
        # print('========')
        # print(output.size())
        # print(output[0,0,0])
        output = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        # print(output[0,0,0])
        # print('========')

        # time.sleep(1)

        # print(output.size())

        if self.bias is not None:
            output += self.bias.unsqueeze(1).unsqueeze(1)

        return output