class Camera:
    def __init__(self, 
                position,
                look_at,
                up,
                f_dist,
                fov
                ):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.f_dist = f_dist
        self._fov = fov #f_dist/half_height
