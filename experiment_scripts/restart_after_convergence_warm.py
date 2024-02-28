import time
from multiprocessing import RawArray

import numpy as np
import toml
from astropy.io import fits
from sklearn.linear_model import ElasticNet

from overlappogram.inversion_field_angles import Inversion

ALPHA = 0.1

if __name__ == '__main__':
    with open("img_norm.toml") as f:
        config = toml.load(f)

    response_cube = fits.getdata(config['paths']['response'])
    inversion = Inversion(rsp_func_cube_file=config['paths']['response'],
                          rsp_dep_name=config['settings']['response_dependency_name'],
                          rsp_dep_list=config['settings']['response_dependency_list'],
                          solution_fov_width=config['settings']['solution_fov_width'],
                          smooth_over=config['settings']['smooth_over'],
                          field_angle_range=config['settings']['field_angle_range'])
    inversion.initialize_input_data(config['paths']['image'],
                                    None,
                                    config['paths']['weights'])

    # fits.writeto("response_matrix.fits", inversion.get_response_function())
    X_d = inversion.get_response_function()
    X_shape = inversion.get_response_function().shape
    X = RawArray('d', X_shape[0] * X_shape[1])
    X_np = np.frombuffer(X).reshape(X_shape)
    np.copyto(X_np, inversion.get_response_function())

    overlappogram_d = fits.getdata(config['paths']['image'])
    overlappogram_shape = overlappogram_d.shape
    overlappogram = RawArray('d', overlappogram_shape[0] * overlappogram_shape[1])
    overlappogram_np = np.frombuffer(overlappogram).reshape(overlappogram_shape)
    np.copyto(overlappogram_np, overlappogram_d)

    weights_d = fits.getdata(config['paths']['weights'])
    weights_shape = weights_d.shape
    weights = RawArray('d', weights_shape[0] * weights_shape[1])
    weights_np = np.frombuffer(weights).reshape(weights_shape)
    np.copyto(weights_np, weights_d)

    enet_model = ElasticNet(alpha=ALPHA,
                            l1_ratio=0.1,
                            max_iter=10_000,
                            precompute=False,
                            positive=True,
                            copy_X=False,
                            fit_intercept=False,
                            warm_start=True,
                            selection='cyclic')

    i = 700
    start = time.time()
    enet_model.fit(X_d, overlappogram_d[i, :], sample_weight=weights_d[i, :])
    data_out = enet_model.predict(X_d)
    em = enet_model.coef_
    end = time.time()
    print("Inversion Time =", end - start)

    # start = time.time()
    # enet_model.fit(X_d, overlappogram_d[i, :], sample_weight=weights_d[i, :])
    # data_out = enet_model.predict(X_d)
    # em = enet_model.coef_
    # end = time.time()
    # print("Reconvergence Time =", end - start)
