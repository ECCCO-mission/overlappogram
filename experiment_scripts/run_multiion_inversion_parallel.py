import argparse
import concurrent.futures
import time
from multiprocessing import RawArray

import numpy as np
import toml
from astropy.io import fits
from sklearn.linear_model import ElasticNet

from overlappogram.inversion_field_angles import Inversion

ALPHA = 5
SELECTION = "cyclic"

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

with open("img_norm.toml") as f:
    config = toml.load(f)

response_cube = fits.getdata(config["paths"]["response"])
inversion = Inversion(
    rsp_func_cube_file=config["paths"]["response"],
    rsp_dep_name=config["settings"]["response_dependency_name"],
    rsp_dep_list=config["settings"]["response_dependency_list"],
    solution_fov_width=config["settings"]["solution_fov_width"],
    smooth_over=config["settings"]["smooth_over"],
    field_angle_range=config["settings"]["field_angle_range"],
)
inversion.initialize_input_data(
    config["paths"]["image"], None, config["paths"]["weights"]
)

# fits.writeto("response_matrix.fits", inversion.get_response_function())
X_d = inversion.get_response_function()
X_shape = inversion.get_response_function().shape
X = RawArray("d", X_shape[0] * X_shape[1])
X_np = np.frombuffer(X).reshape(X_shape)
np.copyto(X_np, inversion.get_response_function())

overlappogram_d = fits.getdata(config["paths"]["image"])
overlappogram_shape = overlappogram_d.shape
overlappogram = RawArray("d", overlappogram_shape[0] * overlappogram_shape[1])
overlappogram_np = np.frombuffer(overlappogram).reshape(overlappogram_shape)
np.copyto(overlappogram_np, overlappogram_d)

weights_d = fits.getdata(config["paths"]["weights"])
weights_shape = weights_d.shape
weights = RawArray("d", weights_shape[0] * weights_shape[1])
weights_np = np.frombuffer(weights).reshape(weights_shape)
np.copyto(weights_np, weights_d)


def init_worker(X, X_shape, overlappogram, overlappogram_shape, weights, weights_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict["X"] = X
    var_dict["X_shape"] = X_shape
    var_dict["overlappogram"] = overlappogram
    var_dict["overlappogram_shape"] = overlappogram_shape
    var_dict["weights"] = weights
    var_dict["weights_shape"] = weights_shape


def init_worker2():
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict["X"] = X
    var_dict["X_shape"] = X_shape
    var_dict["overlappogram"] = overlappogram
    var_dict["overlappogram_shape"] = overlappogram_shape
    var_dict["weights"] = weights
    var_dict["weights_shape"] = weights_shape


def worker_func(i):
    print(i)
    # Simply computes the sum of the i-th row of the input matrix X
    response_matrix = np.frombuffer(var_dict["X"]).reshape(var_dict["X_shape"])
    overlappogram = np.frombuffer(var_dict["overlappogram"]).reshape(
        var_dict["overlappogram_shape"]
    )
    weights = np.frombuffer(var_dict["weights"]).reshape(var_dict["weights_shape"])

    enet_model = ElasticNet(
        alpha=ALPHA,
        l1_ratio=0.1,
        max_iter=10_000,
        precompute=False,
        positive=True,
        copy_X=False,
        fit_intercept=False,
        selection=SELECTION,
    )
    enet_model.fit(response_matrix, overlappogram[i, :], sample_weight=weights[i, :])
    data_out = enet_model.predict(response_matrix)
    em = enet_model.coef_
    return em, data_out


def worker_func_no_init(i):
    print(i)
    # Simply computes the sum of the i-th row of the input matrix X
    #
    # response_matrix = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    # overlappogram = np.frombuffer(var_dict['overlappogram']).reshape(var_dict['overlappogram_shape'])
    # weights = np.frombuffer(var_dict['weights']).reshape(var_dict['weights_shape'])

    enet_model = ElasticNet(
        alpha=ALPHA,
        l1_ratio=0.1,
        max_iter=10_000,
        precompute=False,
        positive=True,
        copy_X=False,
        fit_intercept=False,
        selection=SELECTION,
    )

    enet_model.fit(X_d, overlappogram_d[i, :], sample_weight=weights_d[i, :])
    data_out = enet_model.predict(X_d)
    em = enet_model.coef_
    return em, data_out
    # # create an instance of the GLM class
    # glm = GLM(distr='poisson', score_metric='deviance', fit_intercept=False, alpha=0.1, reg_lambda=5)
    #
    # # fit the model on the training data
    # yhat = glm.fit_predict(response_matrix, overlappogram[i, :])
    #
    # return glm.beta_, yhat


def worker_func_no_init_pass(i, X_d, overlappogram_d, weights_d):
    print(i)
    # Simply computes the sum of the i-th row of the input matrix X
    #
    # response_matrix = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    # overlappogram = np.frombuffer(var_dict['overlappogram']).reshape(var_dict['overlappogram_shape'])
    # weights = np.frombuffer(var_dict['weights']).reshape(var_dict['weights_shape'])

    enet_model = ElasticNet(
        alpha=ALPHA,
        l1_ratio=0.1,
        max_iter=10_000,
        precompute=False,
        positive=True,
        copy_X=False,
        fit_intercept=False,
        selection=SELECTION,
    )

    enet_model.fit(X_d, overlappogram_d[i, :], sample_weight=weights_d[i, :])
    data_out = enet_model.predict(X_d)
    em = enet_model.coef_
    return em, data_out


# We need this check for Windows to prevent infinitely spawning new child
# processes.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverts overlappograms")

    parser.add_argument("config")
    args = parser.parse_args()

    # read config
    # with open(args.config) as f:
    #     config = toml.load(f)
    #
    # response_cube = fits.getdata(config['paths']['response'])
    # inversion = Inversion(rsp_func_cube_file=config['paths']['response'],
    #                       rsp_dep_name=config['settings']['response_dependency_name'],
    #                       rsp_dep_list=config['settings']['response_dependency_list'],
    #                       solution_fov_width=config['settings']['solution_fov_width'],
    #                       smooth_over=config['settings']['smooth_over'],
    #                       field_angle_range=config['settings']['field_angle_range'])
    # inversion.initialize_input_data(config['paths']['image'],
    #                                 None,
    #                                 config['paths']['weights'])
    #
    # # fits.writeto("response_matrix.fits", inversion.get_response_function())
    # X_d = inversion.get_response_function()
    # X_shape = inversion.get_response_function().shape
    # X = RawArray('d', X_shape[0] * X_shape[1])
    # X_np = np.frombuffer(X).reshape(X_shape)
    # np.copyto(X_np, inversion.get_response_function())
    #
    # overlappogram_d = fits.getdata(config['paths']['image'])
    # overlappogram_shape = overlappogram_d.shape
    # overlappogram = RawArray('d', overlappogram_shape[0] * overlappogram_shape[1])
    # overlappogram_np = np.frombuffer(overlappogram).reshape(overlappogram_shape)
    # np.copyto(overlappogram_np, overlappogram_d)
    #
    # weights_d = fits.getdata(config['paths']['weights'])
    # weights_shape = weights_d.shape
    # weights = RawArray('d', weights_shape[0] * weights_shape[1])
    # weights_np = np.frombuffer(weights).reshape(weights_shape)
    # np.copyto(weights_np, weights_d)
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    initargs = (X, X_shape, overlappogram, overlappogram_shape, weights, weights_shape)

    start = time.time()
    # with Pool(processes=11, initializer=init_worker, initargs=initargs) as pool:
    # with ThreadPool(11, initializer=init_worker, initargs=initargs) as pool:
    #     result = pool.map(worker_func, range(config['settings']['detector_row_range'][0],
    #                                          config['settings']['detector_row_range'][1]))

    with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=11,
        #                                             initializer=init_worker,
        #                                             initargs=(X, X_shape, overlappogram,
        #                                             overlappogram_shape, weights, weights_shape)) as executor:
        #     # Start the load operations and mark each future with its URL
        # future_to_url = [executor.submit(worker_func, row)
        #                  for row in range(config['settings']['detector_row_range'][0],
        #                                   config['settings']['detector_row_range'][1])]
        # for future in concurrent.futures.as_completed(future_to_url):
        #     future.result()
        # out = executor.map(worker_func_no_init,  range(config['settings']['detector_row_range'][0],
        #                                       config['settings']['detector_row_range'][1]))
        future_to_url = [
            executor.submit(worker_func_no_init, row)
            for row in range(
                config["settings"]["detector_row_range"][0],
                config["settings"]["detector_row_range"][1],
            )
        ]
        for future in concurrent.futures.as_completed(future_to_url):
            future.result()

    end = time.time()
    print("Inversion Time =", end - start)
