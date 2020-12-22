from subprocess import call
import torch
import pykay
import kay
import random
from tqdm import tqdm
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

objs = []
bunny = pykay.OBJ("./models/bunny/", "bunny1.obj")
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
Ks = torch.Tensor([0, 0, 1.0]).to(device)
Ks.requires_grad = True
LightColor = torch.Tensor([1.0, 1.0, 1.0]).to(device)
light_dir = pykay.normalize(torch.Tensor([0, -1, -1]).to(device))
Shininess = torch.Tensor([10.0]).to(device)
Shininess.requires_grad = True


def shade(geo_id, prim_id, p, wo):
    hit_obj = objs[geo_id]
    n = torch.Tensor(
        [hit_obj.normals[3*prim_id],
         hit_obj.normals[3*prim_id+1],
         hit_obj.normals[3*prim_id+2]]).to(device)
    h = pykay.normalize(-light_dir-p.to(device)).to(device)
    t = n.dot(h)
    x = torch.sign(t) * torch.pow(torch.abs(t), Shininess)
    return Ks*LightColor*(x)


ssn = 1
pic_res = 300
image = torch.zeros(pic_res, pic_res, 3, device=device)


def render():
    global image
    image = torch.zeros(pic_res, pic_res, 3, device=device)
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
                    image[i][j] += shade(rc.geo_id, rc.prim_id,
                                         pixel_pos+dir*rc.dist, dir)
                else:
                    # env background color
                    image[i][j] += torch.Tensor([0, 0, 0.0]
                                                ).requires_grad_(True).to(device)

            image[i][j] /= ssn
    return image


image = render()
target = pykay.imread(
    "/home/simonforgot/proj/Kay/test/results/01_test/target.png").to(device)
diff = torch.abs(target - image)
pykay.imwrite(diff.cpu(), 'results/01_test/init_diff.png')
pykay.imwrite(image.cpu(), 'results/01_test/image.png')


optimizer = torch.optim.Adam([Ks, Shininess], lr=0.25)
# Run 200 Adam iterations.
for t in range(2):
    print('iteration:', t)
    optimizer.zero_grad()
    image = render()
    # Save the intermediate render.
    pykay.imwrite(
        image.cpu(), 'results/optimize_01_test/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (image - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients of the three vertices.
    print('grad:', Ks.grad, Shininess.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.
    print('Ks:', Ks)
    print('Shininess', Shininess)

 
