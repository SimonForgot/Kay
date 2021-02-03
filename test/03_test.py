from subprocess import call
import torch
import pykay
import kay
import random
from tqdm import tqdm

render = pykay.RenderFunction.apply
#

#shape=torch.tensor([[50, 25.0], [200.0, 200.0], [15.0, 150.0],
#[200.0, 50.0], [150.0, 250.0], [50.0, 100.0]],requires_grad = True)     
shape=torch.tensor([[50, 25.0,0,1], [200.0, 200.0,0,1], [15.0, 150.0,0,1],
[200.0, 50.0,0,1], [150.0, 250.0,0,1], [50.0, 100.0,0,1]],requires_grad = True) 


M=torch.tensor([[1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,0,1]],dtype=torch.float32)

print(M.type())
V=torch.tensor([[1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,0,1]],dtype=torch.float32)

P=torch.tensor([[1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,0,1]],dtype=torch.float32)

indices=torch.tensor([[0,1,2],[3,4,5]], dtype = torch.int32)
color=torch.tensor([[0.3,0.5,0.3], [0.3,0.3,0.5]])

t_shape=shape@M@V@P
print(t_shape)
target = render(t_shape,6,indices,2,color)
pykay.imwrite(target.cpu(), 'results/03_test/target.png')


