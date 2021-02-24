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
        
        normals=[]
        colors=[]
        for i in indix:
            p=[]
            for j in range(3):
                p.append(mv_shape[i[j]])
            e0=(p[1]-p[0])[0:3]
            e1=(p[2]-p[1])[0:3]
            n=torch.cross(e0,e1)
            n=pykay.normalize(n)
            #shade
            temp=torch.tensor([0.0,n.dot(torch.tensor([0.0,1.0,0.0]))])
            c=torch.max(temp)
            colors.append(c*torch.tensor([1.0,1.0,1.0]))
            normals.append(n)
        normals=torch.stack(normals)
        colors=torch.stack(colors)
        
        start = time.time()
        #C++ renderer
        kay.render(kay.float_ptr(mv_shape.data_ptr()), 
                p_num,
                kay.unsigned_int_ptr(indix.data_ptr()),
                tri_num,
                f_dist,  
                pic_size,
                kay.float_ptr(colors.data_ptr()),
                kay.float_ptr(image.data_ptr()))

        time_elapsed = time.time() - start
        print('Forward pass, time: %.5f s' % time_elapsed)
        #parameters pass
        ctx.shape = shape
        ctx.p_num = p_num
        ctx.indix = indix
        ctx.tri_num = tri_num
        ctx.normals=normals#later use
        ctx.colors=colors#later use
        ctx.pic_size=pic_size
        ctx.mv_shape=mv_shape
        ctx.f_dist=f_dist
        return image


#give up back faces
    @staticmethod
    def backward(ctx, grad_img):
        shape = ctx.shape
        mv_shape=ctx.mv_shape
        p_num = ctx.p_num
        indix = ctx.indix
        tri_num = ctx.tri_num
        normals=ctx.normals
        colors=ctx.colors
        f_dist=ctx.f_dist
        pic_size=ctx.pic_size
        
        d_shape=torch.zeros(p_num, 4)
        
        kay.d_render(kay.float_ptr(shape.data_ptr()), 
                kay.float_ptr(mv_shape.data_ptr()), 
                p_num,
                kay.unsigned_int_ptr(indix.data_ptr()),
                tri_num,
                f_dist, 
                pic_size,
                kay.float_ptr(grad_img.data_ptr()),
                kay.float_ptr(normals.data_ptr()),
                kay.float_ptr(colors.data_ptr()),
                kay.float_ptr(d_shape.data_ptr()))
        return tuple([d_shape,None,None,None,None,None,None])
