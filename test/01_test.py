import torch
import pykay
import kay
from tqdm import tqdm
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
"""
x=torch.Tensor([4])
x.requires_grad=True

y=torch.Tensor([2])
y.requires_grad=True


z=2*x+y
print(z)
for i in tqdm(range(int(1e3))):
    z=torch.cat((z,2*z[i]+y),0)

z[int(1e3)].backward()
print(x.grad,y.grad)
"""


vertex=torch.Tensor([0,0,0,1,0,0,0,1,0])
index=torch.IntTensor([0,1,2])

rt=kay.Rtcore(kay.float_ptr(vertex.data_ptr()),
            kay.unsigned_int_ptr(index.data_ptr()))


c=pykay.Camera(torch.Tensor([0,0,5]),
                [0,0,-1],[0,1,0],1.0,1.0)
pic_res=300


for i in pic_res:
    for j in pic_res:

rc=rt.intersect()

print(rc.dist,rc.geo_id)

