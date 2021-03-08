import torch
import pykay
import kay
from visdom import Visdom

render = pykay.RenderFunction.apply

pic_size=256
#shape=torch.tensor([[100.0, 100.0,0,1],[7, 75.0,0,1],[-100, -100,0,1],  
#[25.0, 50.0,0,1], [100.0, 25.0,0,1], [75.0, 125.0,0,1]],requires_grad = True) 
cube = pykay.OBJ("./models/02/",  "bunny_big.obj")#""triangle.obj"  "bunny_big.obj"
#obj info load
v_num=cube.vcount
f_num=cube.fcount
print(v_num,f_num)

ones=torch.ones(v_num,1)
shape=torch.tensor(cube.vertices).view(v_num,3)
shape=torch.cat((shape,ones),1)
#transpose
shape=torch.transpose(shape,0,1)

indices=torch.tensor(cube.faces).view(f_num,3)

#right hand coodinate
#put obj near zero point
arg=torch.tensor([0.0],requires_grad=True)
aux= torch.tensor([[1,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]],dtype=torch.float32)
M=torch.tensor([[1,0.,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]],dtype=torch.float32)+arg*aux
eye_pos=50
V=torch.tensor([[1,0,0,0],
    [0,1,0,0],
    [0,0,1,-eye_pos],
    [0,0,0,1]],dtype=torch.float32)
n=-50.0
f=-100.0
#real view volume z(0,-50)
r=pic_size/2
t=pic_size/2
P=torch.tensor([[n/r,0,0,0],
    [0,n/t,0,0],
    [0,0,(n+f)/(n-f),(2*f*n)/(f-n)],
    [0,0,1,0]],dtype=torch.float32)
#mv_shape needs to be passed & used  for  importance sampling

m_shape=M@shape
mv_shape=V@m_shape
mvp_shape=P@mv_shape
w=torch.transpose(mvp_shape,0,1)[:,3]
h_shape=mvp_shape/w
hh_shape=torch.transpose(h_shape,0,1)

f_shape=hh_shape.mul(torch.tensor([1.,-1,1,1]))+torch.tensor([1.,1,0,0])
#used to be derivated
t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])

pass_shape=torch.transpose(mv_shape,0,1)

target = render(t_shape,v_num,indices,f_num,pass_shape,-n,pic_size)
pykay.imwrite(target.cpu(), 'results/03_test/target.png')
target = pykay.imread("results/03_test/target.png")
#_______________________________________________________________________
#perturb
arg1=torch.tensor([-0.3],requires_grad=True)
arg2=torch.tensor([-0.3],requires_grad=True)
aux1= torch.tensor([[1,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]],dtype=torch.float32)
aux2= torch.tensor([[0,0,0,100],
    [0,0,0,100],
    [0,0,0,0],
    [0,0,0,0]],dtype=torch.float32)
M=torch.tensor([[1,0.,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]],dtype=torch.float32)+arg1*aux1+arg2*aux2
m_shape=M@shape
mv_shape=V@m_shape
mvp_shape=P@mv_shape
w=torch.transpose(mvp_shape,0,1)[:,3]

h_shape=mvp_shape/w
hh_shape=torch.transpose(h_shape,0,1)

f_shape=hh_shape.mul(torch.tensor([1.,-1,1,1]))+torch.tensor([1.,1,0,0])
t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])

pass_shape=torch.transpose(mv_shape,0,1)
img= render(t_shape,v_num,indices,f_num,pass_shape,-n,pic_size)
pykay.imwrite(img.cpu(), 'results/03_test/img.png')

diff = torch.abs(target - img)
pykay.imwrite(diff.cpu(), 'results/03_test/init_diff.png')

loss = (img - target).pow(2).sum()
print('loss:', loss.item())
loss.backward(retain_graph=True)
print('grad:', arg1.grad,arg2.grad)

optimizer = torch.optim.RMSprop([arg1,arg2], lr=0.001)

its=700
#250 -1 -1
#500 -1 yes -1 no
#750 650 yes -1 no
#1k 507 yes  -1 no
#1k5 476 -1
#2k 456 yes 475 no
#5k 566 yes 494 no
#5w 571 yes  586 no
viz = Visdom() 
viz.line([[0.]], [0], win='train1', opts=dict(title='grad1'))
viz.line([[0.]], [0], win='train2', opts=dict(title='grad2'))
for t in range(its):
    print('iteration:', t)
    optimizer.zero_grad()

    M=torch.tensor([[1,0.,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]],dtype=torch.float32)+arg1*aux1+arg2*aux2
    m_shape=M@shape
    mv_shape=V@m_shape
    mvp_shape=P@mv_shape
    w=torch.transpose(mvp_shape,0,1)[:,3]
    h_shape=mvp_shape/w
    hh_shape=torch.transpose(h_shape,0,1)
    
    f_shape=hh_shape.mul(torch.tensor([1.,-1,1,1]))+torch.tensor([1.,1,0,0])
    t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])

    pass_shape=torch.transpose(mv_shape,0,1)

    img= render(t_shape,v_num,indices,f_num,pass_shape,-n,pic_size)
    # Save the intermediate render.
    pykay.imwrite(img.cpu(), 'results/03_test_optimize/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())
    #if loss<1:
    #    break
    loss.backward(retain_graph=True)#retain_graph=True
    print('grad:', arg1.grad,arg2.grad)

    viz.line([[arg1.grad]], [t], win='train1', update='append')
    viz.line([[arg2.grad]], [t], win='train2', update='append')
    optimizer.step()
    print('arg:', arg1,arg2)

if its>0:
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/03_test_optimize/iter_%d.png", "-vb", "20M",
        "results/03_test/out.mp4"])
