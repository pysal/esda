# From Carillo
# 1. estimate the probability that an observation is in time 0 using
#    P(T=t) = N_t/(\sum_k N_k)
# 2. estimate P(T=t0 | X=x) with dependent variate is_t0 using all data.
#    should be able to give a predicted probabilty T=t0 for any X
# 3. only for observations where T=t1, compute the probability that T=t0. This is P(T=t0 | X=x)
#    This should be the predicted probability for observations in t1 using the model from before. 
# 4. Then, compute the weights using the tau swap function:
# tau t1 -> t0 := ( P(T=t0|X=x) / (1 - P(T=t0|X=x) ) / (P(T=t0)/(1-P(T=t0)))
# 5. Re-weight the distribution of T1 using tau weights. This is the new distribution in T=1

# In plain english, this re-weights the observed pattern in T1 using the odds of seeing x in t0. 

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class Spatial_Counterfactual(BaseEstimator, TransformerMixin):
    def __init__(self, y0=None, exog0=None, geometry=None, predictor=LogisticRegression):
        self.y0 = y0
        self.exog0 = exog0
        self.geometry = geometry
        self.predictor = predictor
    
    def fit(self, y, X, *, **predictor_kwargs):
        
        # 1
        n0, n1 = len(self.y0), len(y)
        p0, p1 = n0/(n0 + n1), n1/(n0 + n1)
        assert self.exog0.shape[0] == n0, "exog0 and y0 are not aligned!"
        assert X.shape[0] == n1, "exog1 and y1 are not aligned!"
        # is this necessary:
        # assert n0 == n1, "spatial support changes between time periods!"
        # 2
        y_pooled = numpy.hstack((self.y0, y))
        exog_pooled = numpy.row_stack((self.exog0, X))
        is_t0 = numpy.hstack((numpy.ones_like((y0)), numpy.zeros_like((y))))

        self.predictor_ = self.predictor(**predictor_kwards).fit(is_t0, exog_pooled)
        # 3
        t1_p = self.predictor_.predict(X)
        # 4
        self.tau_ = (
                (t1_p/(1 - t1_p))
                /
                (p0 / (1 - p0))
                )
        # 5
        self.actual_ = y
        self.counterfactual_ = self.tau_ * self.actual_

    def predict(self, X, *,):
        n0, n1 = len(self.y0), X.shape[0]
        p0, p1 = n0/(n0 + n1), n1/(n0 + n1)
        tk_p = self.predictor_.predict(X)
        tau_ = (
                (tk_p/(1 - tk_p))
                /
                (p0 / (1 - p0))
                )
        return self.actual_ * tau_
    

