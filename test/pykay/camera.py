import torch
import pykay


class Camera:
    def __init__(self, 
                pos,
                look,
                up,
                f_dist,
                fov
                ):
        self.pos = pos
        self.look_dir = pykay.normalize(look - pos)
        right = pykay.normalize(torch.cross(self.look_dir, pykay.normalize(up)))
        self.up = pykay.normalize(torch.cross(right, self.look_dir))
        self.f_dist = f_dist
        self._fov = fov #half_height/f_dist

