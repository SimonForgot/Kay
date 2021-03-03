#pragma once
#include "ptr.h"

void render(ptr<float> shape,
            int p_num,
            ptr<unsigned int> indices,
            int tri_num,
            float f_dist,
            int pic_res,
            ptr<float> colors,
			ptr<float> rendered_image);

void d_render(ptr<float> shape,
            ptr<float> mv_shape,
            int p_num,
            ptr<unsigned int> indices,
            int tri_num,
            float f_dist,
            int pic_res,
            ptr<float> grad_img,
            ptr<float> normals,
            ptr<float> colors,
			ptr<float> d_shape);