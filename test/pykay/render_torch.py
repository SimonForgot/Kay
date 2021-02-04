import torch
import pykay
import kay
import time


class RenderFunction(torch.autograd.Function):
    """
        The PyTorch interface of C++ kay.
    """
    @staticmethod
    def forward(ctx, shape,p_num,indix,tri_num,color):
        pic_res = 256
        image = torch.zeros(pic_res, pic_res, 3)
        start = time.time()

        kay.render(kay.float_ptr(shape.data_ptr()), 
                p_num,
                kay.unsigned_int_ptr(indix.data_ptr()),
                tri_num,
                kay.float_ptr(color.data_ptr()), 
                kay.float_ptr(image.data_ptr()))

        time_elapsed = time.time() - start
        print('Forward pass, time: %.5f s' % time_elapsed)
        ctx.shape = shape
        ctx.p_num = p_num
        ctx.indix = indix
        ctx.tri_num = tri_num
        ctx.color = color
        return image

    @staticmethod
    def backward(ctx, grad_img):
        shape = ctx.shape
        p_num = ctx.p_num
        indix = ctx.indix
        tri_num = ctx.tri_num
        color = ctx.color
        d_shape=torch.zeros(p_num, 4)
        kay.d_render(kay.float_ptr(shape.data_ptr()), 
                p_num,
                kay.unsigned_int_ptr(indix.data_ptr()),
                tri_num,
                kay.float_ptr(color.data_ptr()),
                kay.float_ptr(grad_img.data_ptr()),
                kay.float_ptr(d_shape.data_ptr()))
        return tuple([d_shape,None,None,None,None])
