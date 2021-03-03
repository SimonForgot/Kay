import re
import os
import torch
import pykay

class OBJ:
    def __init__(self, fdir, filename):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.faces = []
        self.normals = []
        self.vcount = 0
        self.fcount = 0
        self.light = torch.Tensor([0, 0, 0]).requires_grad_(
            False)  # by default,no light
        for line in open(fdir+filename, "r"):
            if line.startswith('#'):
                continue
            if line.startswith('o'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                self.vertices.append(float(values[1]))
                self.vertices.append(float(values[2]))
                self.vertices.append(float(values[3]))
                self.vcount += 1
            elif values[0] == 'f':
                self.faces.append(int(values[1])-1)
                self.faces.append(int(values[2])-1)
                self.faces.append(int(values[3])-1)
                self.fcount += 1
        for i in range(self.fcount):
            face_idx = 3*i
            p0 = [self.vertices[3*(self.faces[face_idx]+0)],
                  self.vertices[3*(self.faces[face_idx]+0)+1],
                  self.vertices[3*(self.faces[face_idx]+0)+2]]
            p1 = [self.vertices[3*(self.faces[face_idx+1])],
                  self.vertices[3*(self.faces[face_idx+1])+1],
                  self.vertices[3*(self.faces[face_idx+1])+2]]
            p2 = [self.vertices[3*(self.faces[face_idx+2])],
                  self.vertices[3*(self.faces[face_idx+2])+1],
                  self.vertices[3*(self.faces[face_idx+2])+2]]
            d0 = torch.Tensor(p1)-torch.Tensor(p0)
            d1 = torch.Tensor(p2)-torch.Tensor(p1)
            n  = pykay.normalize(torch.cross(d0,d1))
            for i in range(3):
                self.normals.append(n[i])
            
