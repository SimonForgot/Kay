import torch
import pykay
import kay
import time


class RenderFunction(torch.autograd.Function):
    """
        The PyTorch interface of C++ kay.
    """
    @staticmethod
    def forward(ctx, shape,p_num,indix,tri_num,mv_shape,f_dist,pic_size):
        image = torch.zeros(pic_size, pic_size, 3)
        #worldwide  direction light,up to down,height=100
        light_dir=pykay.normalize(torch.tensor([0.,-1,0]))
        normals=[]
        for i in indix:
            p=[]
            for j in range(3):
                p.append(mv_shape[i[j]])
            e0=(p[1]-p[0])[0:3]
            e1=(p[2]-p[1])[0:3]
            n=torch.cross(e0,e1)
            n=pykay.normalize(n)
            normals.append(n)
        normals=torch.stack(normals)
        #C++ renderer
        start = time.time()
        kay.render(kay.float_ptr(mv_shape.data_ptr()), 
                p_num,
                kay.unsigned_int_ptr(indix.data_ptr()),
                tri_num,
                f_dist,  
                pic_size,
                kay.float_ptr(image.data_ptr()))
        time_elapsed = time.time() - start
        print('Forward pass, time: %.5f s' % time_elapsed)
        #parameters pass
        ctx.shape = shape
        ctx.p_num = p_num
        ctx.indix = indix
        ctx.tri_num = tri_num
        ctx.color = color
        ctx.normals=normals

        return image

    @staticmethod
    def backward(ctx, grad_img):
        shape = ctx.shape
        p_num = ctx.p_num
        indix = ctx.indix
        tri_num = ctx.tri_num
        color = ctx.color
        ctx.normals=normals
        d_shape=torch.zeros(p_num, 4)

        kay.d_render(kay.float_ptr(shape.data_ptr()), 
                p_num,
                kay.unsigned_int_ptr(indix.data_ptr()),
                tri_num,
                kay.float_ptr(color.data_ptr()),
                kay.float_ptr(grad_img.data_ptr()),
                kay.float_ptr(d_shape.data_ptr()))
        return tuple([d_shape,None,None,None,None,None])
