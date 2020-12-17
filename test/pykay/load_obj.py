import re
import os

class OBJ:
    def __init__(self, fdir, filename):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.faces = []
        self.vcount = 0
        self.fcount = 0 
        for line in open(fdir+filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
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
