import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import rfi.examples.chains as chains
from rfi.explainers.explainer import Explainer
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
from rfi.explanation import Explanation

import logging

logging.basicConfig(level=logging.INFO)

mod1 = linear_model.LinearRegression()
# mod1 = linear_model.Lasso(alpha=0)
# savepath = 'C:/Users/ra59qih/sciebo/LMU/Forschung/Feature_importance/Python/'
savepath = ''

# datasets to use
data = pd.read_csv(savepath + 'extrapolation.csv')

data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']]
data_model = data[['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
ntrain = int(0.7 * data.shape[0])

xcolumns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
ycolumn = ['y']
df_train, df_test = data.iloc[0:ntrain,], data.iloc[ntrain:,]
X_train, y_train = df_train[xcolumns], df_train[ycolumn]
X_test, y_test = df_test[xcolumns], df_test[ycolumn]

# fit models
xcolumns_model = ['x1', 'x2', 'x3', 'x4', 'x5']
df_train_model = data.iloc[0:ntrain,]
X_train_model, y_train_model = df_train_model[xcolumns_model], df_train_model[ycolumn]

mod1.fit(X_train_model, y_train_model)
lasso = linear_model.LassoCV()
lasso.fit(X_train_model, y_train_model)

# mod1.coef_[0, 0] = 0.3
# mod1.coef_[0, 1] = -0.3
# mod1.coef_[0, 2] = 0
# mod1.coef_[0, 3] = 1
# mod1.coef_[0, 4] = 0
# mod1.coef_[0, 5] = 0

"""
scoring = [mean_squared_error, r2_score]
names = ['MSE', 'r2_score']
models = [mod1]
m_names = ['Lasso']

for kk in range(len(models)):
    model = models[kk]
    print('Model: {}'.format(m_names[kk]))
    for jj in np.arange(len(names)):
        print('{}: {}'.format(names[jj],
                              scoring[jj](y_test, model.predict(X_test))))
"""


# explain model

sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)
fsoi = X_train.columns
ordering = [tuple(fsoi)]

wrk = Explainer(mod1.predict, fsoi, X_train,
                loss=mean_squared_error, sampler=sampler,
                decorrelator=decorrelator)


ex_cfi = wrk.ais_via_contextfunc(fsoi, X_test, y_test, context='remainder', marginalize=False)
ex_cfi.hbarplot()
plt.show()

df_cfi = ex_cfi.fi_means_quantiles()
df_cfi['type'] = 'cfi'

ex_pfi = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='remainder', marginalize=False)
ex_pfi.hbarplot()
plt.show()

df_pfi = ex_pfi.fi_means_quantiles()
df_pfi['type'] = 'pfi'

G = ['x1','x3']
ex5 = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='remainder')
ex6 = wrk.dis_from_baselinefunc(G, X_test, y_test, baseline='remainder')
scores_rfi = ex5.scores - ex6.scores
ex_rfi = Explanation(fsoi, scores_rfi)
ex_rfi.hbarplot()
plt.show()

df_rfi = ex_rfi.fi_means_quantiles()
df_rfi['type'] = 'rfi'

df_res = pd.concat([df_pfi, df_rfi, df_cfi]).reset_index()
df_res.to_csv(savepath+'df_res.csv')

print(mod1.coef_, mod1.intercept_, lasso.coef_, lasso.intercept_)