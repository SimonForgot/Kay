import torch
import pykay
import kay

render = pykay.RenderFunction.apply

pic_size=256
#shape=torch.tensor([[100.0, 100.0,0,1],[7, 75.0,0,1],[-100, -100,0,1],  
#[25.0, 50.0,0,1], [100.0, 25.0,0,1], [75.0, 125.0,0,1]],requires_grad = True) 
cube = pykay.OBJ("./models/", "triangle.obj")#"bunny_big.obj"
#obj info load
v_num=cube.vcount
f_num=cube.fcount
print(v_num,f_num)

ones=torch.ones(v_num,1)
shape=torch.tensor(cube.vertices).view(v_num,3)
shape=torch.cat((shape,ones),1)
indices=torch.tensor(cube.faces).view(f_num,3)

#right hand coodinate
M=torch.tensor([[1,0.,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]],requires_grad=True)
eye_pos=50
V=torch.tensor([[1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,-eye_pos,1]],dtype=torch.float32)
n=50.0
f=100.0
#real view volume z(0,-50)
r=pic_size/2
t=pic_size/2
P=torch.tensor([[-n/r,0,0,0],
    [0,-n/t,0,0],
    [0,0,(n+f)/(f-n),1],
    [0,0,(2*f*n)/(f-n),0]],dtype=torch.float32)
#mv_shape needs to be passed & used  for  importance sampling
m_shape=shape@M
mv_shape=m_shape@V
mvp_shape=mv_shape@P
w=mvp_shape[:,3]
h_shape=torch.transpose(m_shape,0,1)/w
hh_shape=torch.transpose(h_shape,0,1)
f_shape=hh_shape.mul(torch.tensor([1.,-1,1,1]))+torch.tensor([1.,1,0,0])
#used to be derivated
t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])

target = render(t_shape,v_num,indices,f_num,mv_shape,n,pic_size)
pykay.imwrite(target.cpu(), 'results/03_test/target.png')
target = pykay.imread("results/03_test/target.png")
#_______________________________________________________________________
#perturb
M=torch.tensor([[0.5,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]],dtype=torch.float32,requires_grad=True)
m_shape=shape@M
mv_shape=m_shape@V
mvp_shape=mv_shape@P
w=m_shape[:,3]
h_shape=torch.transpose(m_shape,0,1)/w
hh_shape=torch.transpose(h_shape,0,1)
f_shape=hh_shape.mul(torch.tensor([1.,-1,1,1]))+torch.tensor([1.,1,0,0])
t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])

img= render(t_shape,v_num,indices,f_num,mv_shape,n,pic_size)
pykay.imwrite(img.cpu(), 'results/03_test/img.png')

diff = torch.abs(target - img)
pykay.imwrite(diff.cpu(), 'results/03_test/init_diff.png')

loss = (img - target).pow(2).sum()
print('loss:', loss.item())
loss.backward(retain_graph=True)
print('grad:', M.grad)

optimizer = torch.optim.Adam([M], lr=0.01)

its=0
# Run 200 Adam iterations.
for t in range(its):
    print('iteration:', t)
    optimizer.zero_grad()

    m_shape=shape@M
    mv_shape=m_shape@V
    mvp_shape=mv_shape@P
    w=m_shape[:,3]
    h_shape=torch.transpose(m_shape,0,1)/w
    hh_shape=torch.transpose(h_shape,0,1)
    f_shape=hh_shape.mul(torch.tensor([1.,-1,1,1]))+torch.tensor([1.,1,0,0])
    t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])

    img= render(t_shape,v_num,indices,f_num,mv_shape,n,pic_size)
    # Save the intermediate render.
    pykay.imwrite(img.cpu(), 'results/03_test_optimize/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())
    # Backpropagate the gradients.
    loss.backward(retain_graph=True)#retain_graph=True
    print('grad:', M.grad)
    optimizer.step()
    print('variable:', M)

if its>0:
    from subprocess import call
    call(["ffmpeg", "-framerate", "10", "-i",
        "results/03_test_optimize/iter_%d.png", "-vb", "20M",
        "results/03_test/out.mp4"])
