from subprocess import call
import torch
import pykay
import kay
import random
from tqdm import tqdm

render = pykay.RenderFunction.apply

#shape=torch.tensor([[50, 25.0], [200.0, 200.0], [15.0, 150.0],
#[200.0, 50.0], [150.0, 250.0], [50.0, 100.0]],requires_grad = True)     
shape=torch.tensor([[50, 25.0], [200.0, 200.0], [15.0, 150.0],
[200.0, 50.0], [150.0, 250.0], [50.0, 100.0]],requires_grad = True) 


indices=torch.tensor([[0,1,2],[3,4,5]], dtype = torch.int32)
color=torch.tensor([[0.3,0.5,0.3], [0.3,0.3,0.5]])


target = render(shape,6,indices,2,color)
pykay.imwrite(target.cpu(), 'results/02_test/target.png')
pykay.imwrite(target.cpu(), 'results/02_test_optimize/target.png')

#perturb the shape
shape=torch.tensor([[50, 25.0], [100.0, 200.0], [15.0, 150.0],
[200.0, 15.0], [150.0, 250.0], [50.0, 100.0]],requires_grad = True)     
                

img = render(shape,6,indices,2,color)
pykay.imwrite(img.cpu(), 'results/02_test/init.png')

diff = torch.abs(target - img)
pykay.imwrite(diff.cpu(), 'results/02_test/init_diff.png')

optimizer = torch.optim.Adam([shape], lr=1)

# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    
    img = render(shape,6,indices,2,color)
    # Save the intermediate render.
    pykay.imwrite(img.cpu(), 'results/02_test_optimize/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())
    
    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients of the three vertices.
    print('grad:', shape.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.
    print('vertices:', shape)

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/02_test_optimize/iter_%d.png", "-vb", "20M",
    "results/02_test/out.mp4"])