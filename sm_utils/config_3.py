import os
import os.path as osp
import sys
import numpy as np

class Config:

    flame_shape_params = 100
    flame_expression_params = 50
    face_corr_fname = '/home/sd2/fansiming/data/Motion-X/tomato_represenation/body_models/smplx/SMPL-X__FLAME_vertex_ids.npy'

    def set_additional_args(self, **kwargs):
        names = self.__dict__
        for k, v in kwargs.items():
            names[k] = v


cfg = Config()
