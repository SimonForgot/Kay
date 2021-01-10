#include <random>
#include "ptr.h"

void render(ptr<float> shape,
            int p_num,
            ptr<unsigned int> indices,
            int tri_num,
            ptr<float> color,
			ptr<float> rendered_image);

void d_render(ptr<float> shape,
            int p_num,
            ptr<unsigned int> indices,
            int tri_num,
			ptr<float> d_shape);