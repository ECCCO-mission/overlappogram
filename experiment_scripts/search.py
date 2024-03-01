import numpy as np
import toml
from astropy.io import fits
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

from overlappogram.inversion_field_angles import Inversion

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

X_d = inversion.get_response_function()
X_shape = inversion.get_response_function().shape

overlappogram_d = fits.getdata(config["paths"]["image"])
overlappogram_shape = overlappogram_d.shape

weights_d = fits.getdata(config["paths"]["weights"])
weights_shape = weights_d.shape

elastic = ElasticNet(fit_intercept=False)
# elastic.fit(X_d, overlappogram_d[700, :])

search = GridSearchCV(
    estimator=elastic,
    param_grid={"alpha": np.logspace(-2, 10, 8), "l1_ratio": np.linspace(0.1, 0.9, 8)},
    scoring="r2",
    n_jobs=10,
    refit=False,
    cv=2,
)
search.fit(X_d, overlappogram_d[700, :])
print(search.best_params_)
print(abs(search.best_score_))
