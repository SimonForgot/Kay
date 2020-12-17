import torch
import pykay
import kay
from tqdm import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

objs=[]
bunny=pykay.OBJ("./models/bunny/","bunny.obj")
objs.append(bunny)
tri=pykay.OBJ("./models/","triangle.obj")
objs.append(tri)
#test=pykay.OBJ("/home/simonforgot/proj/Kay/test/models/cornellbox/","left.obj")
#new_ver=[i * 0.001 for i in test.vertices]


rt = kay.Rtcore()
vertex=[]
index=[]
for i in range(len(objs)):
    vertex.append(torch.Tensor(objs[i].vertices))
    index.append(torch.IntTensor(objs[i].faces))
    print(objs[i].vcount,objs[i].fcount)
    rt.addGeo(kay.float_ptr(vertex[i].data_ptr()),
                kay.unsigned_int_ptr(index[i].data_ptr()),
                objs[i].vcount,
                objs[i].fcount)         
rt.RTsetup()
# camera info init
pos = torch.Tensor([0, 0, -0.2])
look = torch.Tensor([0, 0, 0])
up = torch.Tensor([0, 1, 0])
c = pykay.Camera(pos, look, up, 1.0, 1.0)

pic_res = 300
spp = 8
image = torch.ones(pic_res, pic_res, 3, device=device)

center = c.pos+c.f_dist*c.look_dir
temp = c.f_dist*c._fov  # half height
right = torch.cross(c.look_dir, c.up)
left_up = center+temp*c.up-temp*right
"""
def L(count):
    if count!=0:
        res=x*L(count-1)
        res=res+L(count-1)
        return res
    else:
        return 3
"""
for i in tqdm(range(pic_res)):
    for j in range(pic_res):
        dir = left_up - i*2*temp/pic_res*c.up + j*2*temp/pic_res*right
        dir = dir -c.pos
        rc = rt.intersect(c.pos[0], c.pos[1], c.pos[2],
                          dir[0], dir[1], dir[2])
        if rc.hit_flag:
            #print(rc.geo_id)
            image[i][j] = torch.Tensor([1.0, 0.5, 0.5])
        else:
            image[i][j] = torch.Tensor([0.1, 0.2, 0.5])

pykay.imwrite(image.cpu(), 'results/01_test/target.png')

"""
y=torch.Tensor([2])
y.requires_grad=True

l=L(3)
l.backward()
print(x.grad)

c=0
def func(count):
    if count!=0:
        global z
        global c
        z=torch.cat((z,z[c]+x),0)
        c=c+1
        count=count-1
        func(count)
    else:
        return 

func(6)
print(z)
z[0]=torch.stack((z[0],z[0]+x),0)
print(z)

#for i in tqdm(range(2)):

#z[int(1e1)].backward()
#print(x.grad,y.grad)
"""

"""

        #get L
        #if hit
        z[i*pic_res+j]=
rc=rt.intersect()

print(rc.dist,rc.geo_id)

"""
