from dataclasses import dataclass

from sklearn.linear_model import ElasticNet as enet

from overlappogram.abstract_model import AbstractModel


@dataclass(order=True)
class ElasticNetModel(AbstractModel):
    model: enet = enet()

    def invert(self, response_function, data, sample_weights = None):
        #print(sample_weights)
        self.model.fit(response_function, data, sample_weight=sample_weights)
        #self.model.fit(response_function, data)
        #score=(self.model.score(response_function, data, sample_weight=sample_weights))
        data_out = self.model.predict(response_function)
        em = self.model.coef_
        return em, data_out
        #return em, data_out, score

    def add_fits_keywords(self, header):
        params = self.model.get_params()
        #print(params)
        header['INVMDL'] = ('Elastic Net', 'Inversion Model')
        header['ALPHA'] = (params['alpha'], 'Inversion Model Alpha')
        header['RHO'] = (params['l1_ratio'], 'Inversion Model Rho')

    def get_score(self, response_function, data):
        score = self.model.score(response_function, data)
        return score
