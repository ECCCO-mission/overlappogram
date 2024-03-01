import numpy as np
from scipy.optimize import minimize


class Model:
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def get_params(self):
        return {"alpha": self.alpha}


class HomemadeModel:
    def __init__(self, model):
        self.model = model

    def invert(self, response_function, data):
        # weights=np.full((2048),1)

        def f(x):
            s = 0
            # x=congrid(x,response_function[:,0,:].shape)
            for t in range(response_function.shape[0]):
                s += self.model.alpha * np.linalg.norm(
                    x[t]
                )  # np.linalg.norm(response_function[t]@x[t] -data)+self.alpha*np.linalg.norm(R@x)
            return s

        # data_out = self.model.predict(np.full_like(response_function,0.))

        em = minimize(f, np.ones((15, 256), dtype=float))
        print(em)
        data_out = sum([response_function[t] @ em[t] for t in range(15)])
        # print(self.model.intercept_)
        return em, data_out

    def add_fits_keywords(self, header):
        params = self.model.get_params()
        # print(params)
        header["INVMDL"] = ("Elastic Net", "Inversion Model")
        header["ALPHA"] = (params["alpha"], "Inversion Model Alpha")
        # header['RHO'] = (params['l1_ratio'], 'Inversion Model Rho')
