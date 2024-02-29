import os

NUMPY_TREAD_COUNT = 1
os.environ["OMP_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["MKL_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)

import argparse
import os
import time

import toml
from sklearn.linear_model import ElasticNet

from overlappogram.elasticnet_model import ElasticNetModel as model
from overlappogram.inversion_field_angles import Inversion


def run_inversion(config: dict):
    inversion = Inversion(rsp_func_cube_file=config['paths']['response'],
                          rsp_dep_name=config['settings']['response_dependency_name'],
                          rsp_dep_list=config['settings']['response_dependency_list'],
                          solution_fov_width=config['settings']['solution_fov_width'],
                          smooth_over=config['settings']['smooth_over'],
                          field_angle_range=config['settings']['field_angle_range'])

    inversion.initialize_input_data(config['paths']['image'],
                                    None,
                                    config['paths']['weights'])

    for alpha in config['settings']['alphas']:
        for rho in config['settings']['rhos']:
            enet_model = ElasticNet(alpha=alpha,
                                    l1_ratio=rho,
                                    tol=1E-4,
                                    max_iter=10_000,
                                    precompute=False,
                                    positive=True,
                                    copy_X=False,
                                    fit_intercept=False,
                                    selection='cyclic',
                                    warm_start=False)
            inv_model = model(enet_model)

            basename = os.path.splitext(os.path.basename(config['paths']['image']))[0]

            start = time.time()

            postfix = 'x'+str(config['settings']['solution_fov_width'])+'_'+str(rho*10)+'_'+str(alpha)+'_wpsf'
            inversion.multiprocessing_invert(inv_model,
                                             config['paths']['output'],
                                             output_file_prefix=basename,
                                             output_file_postfix=postfix,
                                             detector_row_range=config['settings']['detector_row_range'],
                                             score=True)

            end = time.time()
            print("Inversion Time =", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inverts overlappograms')

    parser.add_argument('config')
    args = parser.parse_args()

    # read config
    with open(args.config) as f:
        config = toml.load(f)

    run_inversion(config)
