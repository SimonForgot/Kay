from subprocess import call
import torch
import pykay
import kay
import random
from tqdm import tqdm

render = pykay.RenderFunction.apply

pic_size=256
shape=torch.tensor([[-100, -100,0,1], [100.0, 100.0,0,1], [7, 75.0,0,1],
[100.0, 25.0,0,1], [75.0, 125.0,0,1], [25.0, 50.0,0,1]],requires_grad = True) 
M=torch.tensor([[1,0.,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]],requires_grad=True)
eye_pos=50
V=torch.tensor([[1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,-eye_pos,1]],dtype=torch.float32)
n=50
f=100
r=pic_size/2
t=pic_size/2
P=torch.tensor([[-n/r,0,0,0],
    [0,-n/t,0,0],
    [0,0,(n+f)/(f-n),1],
    [0,0,(2*f*n)/(f-n),0]],dtype=torch.float32)
m_shape=shape@M@V@P
w=m_shape[:,3]
h_shape=torch.transpose(m_shape,0,1)/w
hh_shape=torch.transpose(h_shape,0,1)
f_shape=hh_shape+torch.tensor([1.,1,0,0])
t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])


indices=torch.tensor([[0,1,2],[3,4,5]], dtype = torch.int32)
color=torch.tensor([[0.3,0.5,0.3], [0.3,0.3,0.5]])
target = render(t_shape,6,indices,2,color)
pykay.imwrite(target.cpu(), 'results/03_test/target.png')



shape=torch.tensor([[-100, -100,0,1], [100.0, 100.0,0,1], [7, 75.0,0,1],
[100.0, 25.0,0,1], [75.0, 125.0,0,1], [25.0, 50.0,0,1]],requires_grad = True) 
#perturb
M=torch.tensor([[0.7,0,0,0],
    [0,0.5,0,0],
    [0,0,1,0],
    [0,-5,0,1]],dtype=torch.float32,requires_grad=True)
m_shape=shape@M@V@P

w=m_shape[:,3]
h_shape=torch.transpose(m_shape,0,1)/w
hh_shape=torch.transpose(h_shape,0,1)
f_shape=hh_shape+torch.tensor([1.,1,0,0])
t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])


img= render(t_shape,6,indices,2,color)
pykay.imwrite(img.cpu(), 'results/03_test/img.png')

diff = torch.abs(target - img)
pykay.imwrite(diff.cpu(), 'results/03_test/init_diff.png')


loss = (img - target).pow(2).sum()
print('loss:', loss.item())
loss.backward(retain_graph=True)
print('grad:', M.grad)


optimizer = torch.optim.Adam([M], lr=0.001)

its=300
# Run 200 Adam iterations.
for t in range(its):
    print('iteration:', t)
    optimizer.zero_grad()
    
    m_shape=shape@M@V@P
    
    w=m_shape[:,3]
    h_shape=torch.transpose(m_shape,0,1)/w
    hh_shape=torch.transpose(h_shape,0,1)
    f_shape=hh_shape+torch.tensor([1.,1,0,0])
    t_shape=f_shape*torch.tensor([pic_size/2,pic_size/2,1.,1])
    img = render(t_shape,6,indices,2,color)
    # Save the intermediate render.
    pykay.imwrite(img.cpu(), 'results/03_test_optimize/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())
    
    # Backpropagate the gradients.
    loss.backward(retain_graph=True)#retain_graph=True
    # Print the gradients of the three vertices.
    print('grad:', M.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.
    print('variable:', M)

if its>0:
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/03_test_optimize/iter_%d.png", "-vb", "20M",
        "results/03_test/out.mp4"])
