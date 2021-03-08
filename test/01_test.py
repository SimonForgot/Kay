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

#PBR 
Ks = torch.Tensor([0.5, 0, 0.0])
Ks.requires_grad = True
Kd = torch.Tensor([0.5, 0, 0.0])
Kd.requires_grad = True
Ka = torch.Tensor([0.1, 0, 0.0])
Ka.requires_grad = True
Ia = torch.Tensor([0.1, 0.1, 0.1])
Ia.requires_grad = True
LightColor = torch.Tensor([100.0, 100.0, 100.0])
LightColor.requires_grad =True
light_dir = pykay.normalize(torch.Tensor([0, -1, 0]))
rough = torch.Tensor([10.0])
rough.requires_grad = True

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

def shade_pbr(geo_id, prim_id, p, wo):#wo=view
    hit_obj = objs[geo_id]
    n = torch.Tensor(
        [hit_obj.normals[3*prim_id],
         hit_obj.normals[3*prim_id+1],
         hit_obj.normals[3*prim_id+2]])
    h = pykay.normalize(-light_dir-wo)
    h_dot_v=h.dot(-wo)
    n_dot_h=n.dot(h)
    n_dot_v=n.dot(-wo)
    n_dot_l=n.dot(-light_dir)
    l_dot_h=h.dot(-light_dir)
    D=torch.exp(torch.Tensor([(n_dot_h*n_dot_h-1)/(rough*rough*n_dot_h*n_dot_h)]))/(4*rough*rough*torch.pow(n_dot_h,4))
    F0=0.03
    F=F0+(1-F0)*torch.pow(h_dot_v, 5)
    Ga=2*n_dot_h*n_dot_v/h_dot_v
    Gb=2*n_dot_h*n_dot_l/l_dot_h
    G=torch.min(torch.Tensor([1,Ga,Gb]))
    d_temp=torch.max(torch.Tensor([n.dot(-light_dir), 0.0]))
    s_temp =0.25*D*F*G/(n.dot(-light_dir)*n.dot(-wo)) 
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
                #dir = pixel_pos - c.pos
                r = pykay.ray(c.pos, dir)
                rc = rt.intersect(r.o[0], r.o[1], r.o[2],
                                  r.d[0], r.d[1], r.d[2])
                if rc.hit_flag:
                    image[i][j] += shade_pbr(rc.geo_id, rc.prim_id,c.pos+dir*rc.dist, dir)*ao_image[i][j]#AO
                else:
                    # env background color
                    image[i][j] += torch.Tensor([0, 0, 0.0]
                                                ).requires_grad_(True)

            image[i][j] /= ssn
    return image


image = render()#33 sec/4 spp  
target = pykay.imread(
    "results/pbr_ao/target.png")#143 sec
diff = torch.abs(target - image)
pykay.imwrite(diff.cpu(), 'results/pbr_ao/init_diff.png')
pykay.imwrite(image.cpu(), 'results/pbr_ao/image.png')


optimizer = torch.optim.Adam([Ks,Kd,Ka,Ia,rough,LightColor], lr=0.1)
# Run 200 Adam iterations.
its=150
for t in range(its):
    print('iteration:', t)
    optimizer.zero_grad()
    image = render()
    # Save the intermediate render.
    pykay.imwrite(
        image.cpu(), 'results/optimize_pbr_ao/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (image - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients of the three vertices.
    print('grad:', Ks.grad, Kd.grad, Ka.grad,Ia.grad,rough.grad,LightColor.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.
    print('Ks:', Ks)
    print('Kd:', Kd)
    print('Ka:', Ka)
    print('Ia:', Ia)
    print('Rough:', rough)
    print('LightColor:',LightColor)

 
if its>0:
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i", "results/optimize_pbr_ao/iter_%d.png", "-vb", "20M","results/pbr_ao/out.mp4"])