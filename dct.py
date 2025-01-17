import torch
import torch.fft as fft
import numpy as np
import scipy.fft as sp_fft
import torch.nn as nn
import copy
device='cuda' if torch.cuda.is_available() else 'cpu'
class DCTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_np = input.detach().cpu().numpy()
        batch_size, channels, height, width = input_np.shape
        block_size = 8
        dct_output = np.zeros_like(input_np)
        for b in range(batch_size):
            for c in range(channels):
                for y in range(0, height, block_size):
                    for x in range(0, width, block_size):
                        block = input_np[b, c, y:y+block_size, x:x+block_size]
                        dct_block = sp_fft.dct(sp_fft.dct(block, norm='ortho', axis=0), norm='ortho', axis=1)
                        dct_output[b, c, y:y+block_size, x:x+block_size] = dct_block
        
        return torch.from_numpy(dct_output).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output_np = grad_output.cpu().numpy()
        batch_size, channels, height, width = grad_output_np.shape
        block_size = 8
        idct_output = np.zeros_like(grad_output_np)
        for b in range(batch_size):
            for c in range(channels):
                for y in range(0, height, block_size):
                    for x in range(0, width, block_size):
                        block = grad_output_np[b, c, y:y+block_size, x:x+block_size]
                        idct_block = sp_fft.idct(sp_fft.idct(block, norm='ortho', axis=0), norm='ortho', axis=1)
                        idct_output[b, c, y:y+block_size, x:x+block_size] = idct_block
        
        return torch.from_numpy(idct_output).to(grad_output.device)

class DCTLayer(nn.Module):
    def forward(self, input):
        return DCTFunction.apply(input)

class IDCTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_np = input.detach().cpu().numpy()
        batch_size, channels, height, width = input_np.shape
        block_size = 8
        idct_output = np.zeros_like(input_np)
        for b in range(batch_size):
            for c in range(channels):
                for y in range(0, height, block_size):
                    for x in range(0, width, block_size):
                        block = input_np[b, c, y:y+block_size, x:x+block_size]
                        idct_block = sp_fft.idct(sp_fft.idct(block, norm='ortho', axis=0), norm='ortho', axis=1)
                        idct_output[b, c, y:y+block_size, x:x+block_size] = idct_block
        
        return torch.from_numpy(idct_output).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output_np = grad_output.cpu().numpy()
        batch_size, channels, height, width = grad_output_np.shape
        block_size = 8
        dct_output = np.zeros_like(grad_output_np)
        for b in range(batch_size):
            for c in range(channels):
                for y in range(0, height, block_size):
                    for x in range(0, width, block_size):
                        block = grad_output_np[b, c, y:y+block_size, x:x+block_size]
                        dct_block = sp_fft.dct(sp_fft.dct(block, norm='ortho', axis=0), norm='ortho', axis=1)
                        dct_output[b, c, y:y+block_size, x:x+block_size] = dct_block
        
        return torch.from_numpy(dct_output).to(grad_output.device)

class IDCTLayer(nn.Module):
    def forward(self, input):
        return IDCTFunction.apply(input)
def create_perturbation_mask(batch_size, channels, height, width, block_size=8, num_elements=20): 
    mask = torch.zeros(batch_size, channels, height, width, dtype=torch.bool, device='cuda')
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_mask = torch.zeros(block_size, block_size, dtype=torch.bool, device='cuda')
            block_mask.view(-1)[-num_elements:] = True
            mask[:, :, y:y+block_size, x:x+block_size] = block_mask
    return mask
def add_perturbation_with_mask(tensor, perturbation, mask):
    tensor = tensor.cuda()
    perturbation = perturbation.cuda()
    perturbed_tensor = torch.where(mask, tensor + perturbation, tensor)
    return perturbed_tensor
class NoiseTransform(): 
    def __init__(self,shape=(32,32)):
        self.mask=torch.ones(shape,dtype=torch.bool) 
    def dct_on_batches(self,images):
        images=images.cpu().detach().numpy()
        c=images.shape[1]
        new_images=[]
        for image in images:
            new_image=np.zeros_like(image)
            for c in range(3):
                block=copy.deepcopy(image[c])
                dct_block=sp_fft.dct(sp_fft.dct(block,norm='ortho',axis=0),norm='ortho',axis=1)
                new_image[c] = dct_block
            new_images.append(torch.tensor(new_image))
        return torch.stack(new_images)
    def idct_on_batches(self,images):
        images=images.cpu().detach().numpy()
        c=images.shape[1]
        new_images=[]
        for image in images:
            new_image=np.zeros_like(image)
            for c in range(0,3):
                block=copy.deepcopy(image[c])
                idct_block=sp_fft.idct(sp_fft.idct(block,norm='ortho',axis=0),norm='ortho',axis=1)
                new_image[c] = idct_block
            new_images.append(torch.tensor(new_image))
        return torch.stack(new_images)
    def add_noise(self,images,noises): #
        new_images=[]
        for image,noise in zip(images,noises):
            new_image=torch.zeros_like(image)
            c_noise=copy.deepcopy(noise)
            for c in range(0,image.shape[0]):
                c_noise[c][~self.mask]=0
                new_image[c]=c_noise[c]+image[c]
            new_images.append(torch.tensor(new_image))
        return torch.stack(new_images)
import torch.nn.functional as F

def dct_2d(x, norm=None):
    """
    2D Discrete Cosine Transform, Type II (a.k.a. the DCT-II), normalized according to `norm`.
    :param x: Input tensor of shape (N, C, H, W).
    :param norm: Normalization mode ('ortho' or None).
    :return: DCT transformed tensor.
    """
    X1 = dct_1d(x, norm=norm)
    X2 = dct_1d(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d(X, norm=None):
    """
    2D Inverse Discrete Cosine Transform, Type III (a.k.a. the IDCT-III), normalized according to `norm`.
    :param X: Input tensor of shape (N, C, H, W).
    :param norm: Normalization mode ('ortho' or None).
    :return: Inverse DCT transformed tensor.
    """
    x1 = idct_1d(X, norm=norm)
    x2 = idct_1d(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def dct_1d(x, norm=None):
    """
    1D Discrete Cosine Transform, Type II (a.k.a. the DCT-II), normalized according to `norm`.
    :param x: Input tensor of shape (N, C, L).
    :param norm: Normalization mode ('ortho' or None).
    :return: DCT transformed tensor.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v, dim=1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def idct_1d(X, norm=None):
    """
    1D Inverse Discrete Cosine Transform, Type III (a.k.a. the IDCT-III), normalized according to `norm`.
    :param X: Input tensor of shape (N, C, L).
    :param norm: Normalization mode ('ortho' or None).
    :return: Inverse DCT transformed tensor.
    """
    X_shape = X.shape
    N = X_shape[-1]

    X = X.contiguous().view(-1, N)

    if norm == 'ortho':
        X[:, 0] *= np.sqrt(N) * 2
        X[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X
    V_t_i = torch.zeros_like(V_t_r, device=X.device)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r[..., ::2], V_r[..., 1::2].flip([1])], dim=-1)

    v = torch.fft.ifft(V, dim=1)
    x = v.real.view(*X_shape)

    return x

def dct_on_batches(images):
    return dct_2d(images, norm='ortho')

def idct_on_batches(images):
    return idct_2d(images, norm='ortho')

class Image_DCT_Transform():
    def __init__(self,shape):
        self.mask=torch.ones(shape,dtype=torch.bool) 
        for x in range(shape[0]):
            for y in range(shape[1]):
                if shape[1]-y<x:
                    self.mask[x][y]=0
    def fft_on_batches(self,images):
        channel=images.shape[1]
        fft_images=[]
        for image in images:
            new_image=torch.rand_like(image)
            for c in range(channel):
                f=torch.fft.fft2(image[c])
                new_image[c]=f.real
            fft_images.append(new_image)
        return torch.stack(fft_images)
    def ifft_on_batches(self,images):
        channel=images.shape[1]
        ifft_images=[]
        for image in images:
            new_image=torch.rand_like(image)
            for c in range(channel):
                f=torch.fft.ifft2(image[c])
                new_image[c]=(torch.abs(f).squeeze())
            ifft_images.append(new_image)

        return torch.stack(ifft_images)
    def add_noise(self,images,noises): #
        new_images=[]
        for image,noise in zip(images,noises):
            new_image=torch.zeros_like(image)
            c_noise=copy.deepcopy(noise)
            for c in range(image.shape[0]):
                c_noise[c][self.mask]=0
                new_image[c]=c_noise[c]+image[c]
            new_images.append(new_image)
        return torch.stack(new_images)