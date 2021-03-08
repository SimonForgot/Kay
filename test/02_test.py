from subprocess import call
import torch
import pykay
import kay
import random
import math
from tqdm import tqdm

pic_res = 256
objs = []
bunny = pykay.OBJ("./models/01/", "bunny.obj")
objs.append(bunny)

rt = kay.Rtcore()
vertex = []
index = []
for i in range(len(objs)):
    vertex.append(torch.Tensor(objs[i].vertices))
    index.append(torch.IntTensor(objs[i].faces))
    print(objs[i].vcount, objs[i].fcount)
    rt.addGeo(kay.float_ptr(vertex[i].data_ptr()),
              kay.unsigned_int_ptr(index[i].data_ptr()),
              objs[i].vcount,
              objs[i].fcount)
rt.RTsetup()
# camera info init
pos = torch.Tensor([0.0, 0.0, 0.2])
look = torch.Tensor([0, 0, 0])
up = torch.Tensor([0, 1, 0])
c = pykay.Camera(pos, look, up, 1.0, 1.0)

center = c.pos+c.f_dist*c.look_dir
temp = c.f_dist*c._fov  # half height
right = torch.cross(c.look_dir, c.up)
left_up = center+temp*c.up-temp*right

# Blinn-Phong Model Shader
Ks = torch.Tensor([0.5, 0, 0.0])
Ks.requires_grad = True
Kd = torch.Tensor([0.4, 0, 0.0])
Kd.requires_grad = True
Ka = torch.Tensor([0.1, 0, 0.0])
Ka.requires_grad = True
Ia = torch.Tensor([1.0, 1.0, 1.0])
Ia.requires_grad = True
LightColor = torch.Tensor([100.0, 100.0, 100.0])
light_dir = pykay.normalize(torch.Tensor([0, -1, 0]))
Shininess = torch.Tensor([10.0])
Shininess.requires_grad = True


ao_image = torch.zeros(pic_res, pic_res, 3)
for i in tqdm(range(pic_res)):
    for j in range(pic_res):
        pixel_pos = left_up - (i+random.random())*2*temp / \
        pic_res*c.up + (j+random.random())*2*temp/pic_res*right
        dir = pykay.normalize(pixel_pos - c.pos)
        r = pykay.ray(c.pos, dir)
        rc = rt.intersect(r.o[0], r.o[1], r.o[2],r.d[0], r.d[1], r.d[2])
        if rc.hit_flag:
            geo_id=rc.geo_id
            prim_id=rc.prim_id
            hit_obj = objs[geo_id]
            n = torch.Tensor([hit_obj.normals[3*prim_id],hit_obj.normals[3*prim_id+1],hit_obj.normals[3*prim_id+2]])
            p=c.pos+dir*(rc.dist-0.00001)
            count=0
            ao_count=0
            while count<64:
                wz=2*random.random()-1
                r=math.sqrt(1-wz*wz)
                phi=2*math.pi*random.random()
                wx=r*math.cos(phi)
                wy=r*math.sin(phi)
                sample_dir=pykay.normalize(torch.Tensor([wx,wy,wz]))
                temp_res=n.dot(sample_dir)
                if temp_res>0:
                    ry = pykay.ray(p, sample_dir)
                    record = rt.intersect(ry.o[0], ry.o[1], ry.o[2],ry.d[0], ry.d[1], ry.d[2])
                    if record.hit_flag:
                        ao_count+=1
                    count+=1
            ao_image[i][j] +=torch.Tensor([1.0,1.0,1.0])*(1-ao_count*1.0/count)
        else:
            ao_image[i][j] += torch.Tensor([0, 0, 0.0]).requires_grad
print("ao_image build")

def shade_blinn_phong(geo_id, prim_id, p, wo):
    hit_obj = objs[geo_id]
    n = torch.Tensor(
        [hit_obj.normals[3*prim_id],
         hit_obj.normals[3*prim_id+1],
         hit_obj.normals[3*prim_id+2]])
    h = pykay.normalize(-light_dir-wo)
    t = n.dot(h)
    d_temp=torch.max(torch.Tensor([n.dot(-light_dir), 0.0]))
    s_temp = torch.pow(torch.max(torch.Tensor([t, 0.0])), Shininess)
    return LightColor/(100-p[1]*p[1])*(Kd*d_temp+Ks*s_temp)
    

ssn = 1
image = torch.zeros(pic_res, pic_res, 3)

def render():
    global image
    image = torch.zeros(pic_res, pic_res, 3)
    for i in tqdm(range(pic_res)):
        for j in range(pic_res):
            for k in range(ssn):
                pixel_pos = left_up - (i+random.random())*2*temp / \
                    pic_res*c.up + (j+random.random())*2*temp/pic_res*right
                dir = pykay.normalize(pixel_pos - c.pos)
                r = pykay.ray(c.pos, dir)
                rc = rt.intersect(r.o[0], r.o[1], r.o[2],
                                  r.d[0], r.d[1], r.d[2])
                if rc.hit_flag:
                    image[i][j] += Ia*ao_image[i][j]#shade_blinn_phong(rc.geo_id, rc.prim_id,pixel_pos+dir*rc.dist, dir)+Ia*Ka#Ka#ao_image[i][j]
                else:
                    image[i][j] += torch.Tensor([0, 0, 0.0]).requires_grad
            image[i][j] /= ssn
    return image


image = render()
#pykay.imwrite(image.cpu(), 'results/blinn_phong/AO.png')
#input()
target = pykay.imread(
    "results/blinn_phong/target.png")
diff = torch.abs(target - image)
pykay.imwrite(diff.cpu(), 'results/blinn_phong/init_diff.png')
pykay.imwrite(image.cpu(), 'results/blinn_phong/image.png')

loss = (image - target).pow(2).sum()
print('loss:', loss.item())
loss.backward()
print('grad:', Ks.grad, Kd.grad, Ka.grad,Ia.grad,Shininess.grad)

optimizer = torch.optim.Adam([Ks,Kd,Ka,Ia,Shininess], lr=0.1)
# Run 200 Adam iterations.
its=200
for t in range(its):
    print('iteration:', t)
    optimizer.zero_grad()
    image = render()
    pykay.imwrite(image.cpu(), 'results/optimize_blinn_phong/iter_{}.png'.format(t))
    loss = (image - target).pow(2).sum()
    print('loss:', loss.item())
    loss.backward()
    print('grad:', Ks.grad, Kd.grad, Ka.grad,Ia.grad,Shininess.grad)
    optimizer.step()
    print('Ks:', Ks)
    print('Kd:', Kd)
    print('Ka:', Ka)
    print('Ia:', Ia)
    print('Shininess:', Shininess)

 
if its>0:
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/optimize_blinn_phong/iter_%d.png", "-vb", "20M",
        "results/blinn_phong/out.mp4"])