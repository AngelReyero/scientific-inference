import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.formula.api as smf

from fippy.explainers import Explainer
from fippy.samplers import GaussianSampler
import logging

logging.basicConfig(level=logging.INFO)

# mod1 = linear_model.LinearRegression()
# mod1 = linear_model.Lasso(alpha=0)
rf = RandomForestRegressor()

# savepath = 'C:/Users/ra59qih/sciebo/LMU/Forschung/Feature_importance/Python/'
# savepath = '~/university/postdoc/research/fi_inference/code/paper_2022_feature_importance_guide/Simulation/Python/'
savepath = ''

# datasets to use
data = pd.read_csv(savepath + 'extrapolation.csv')

data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
ntrain = int(0.7 * data.shape[0])

xcolumns = ['x1', 'x2', 'x3', 'x4', 'x5']
ycolumn = ['y']
df_train, df_test = data.iloc[0:ntrain,], data.iloc[ntrain:,]
X_train, y_train = df_train[xcolumns], df_train[ycolumn]
X_test, y_test = df_test[xcolumns], df_test[ycolumn]

# fit models

# mod1.fit(X_train, y_train)
# mod1 = smf.ols(formula='y ~ x1 + x2 + x3 + np.square(x3) + x4 + np.square(x4) + x5 + np.square(x5) + x3:x4 + x3:x5 + x4:x5', data=df_train).fit()
#mod1.params[9] = 0
#mod1.params[10] = 0.2960963746
rf.fit(X_train, y_train)

scoring = [mean_squared_error, r2_score]
names = ['MSE', 'r2_score']
models = [rf]
m_names = ['LinearRegression']

for kk in range(len(models)):
    model = models[kk]
    print('Model: {}'.format(m_names[kk]))
    for jj in np.arange(len(names)):
        print('{}: {}'.format(names[jj],
                              scoring[jj](y_test, model.predict(X_test))))



# explain model

sampler = GaussianSampler(X_train)
wrk = Explainer(rf.predict, X_train, loss=mean_squared_error, sampler=sampler)

ex_cfi = wrk.cfi(X_test, y_test, nr_runs=50)
ex_cfi.hbarplot()
plt.show()

df_cfi = ex_cfi.fi_means_quantiles()
df_cfi['type'] = 'cfi'


ex_pfi = wrk.pfi(X_test, y_test, nr_runs=50)
ex_pfi.hbarplot()
plt.show()

df_pfi = ex_pfi.fi_means_quantiles()
df_pfi['type'] = 'pfi'


""" G = ['x1','x3']
ex_rfi = wrk.rfi(G, X_test, y_test, nr_resample_marginalize=100)
# ex5 = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='remainder')
# ex6 = wrk.dis_from_baselinefunc(G, X_test, y_test, baseline='remainder')
# scores_rfi = ex5.scores - ex6.scores
# ex_rfi = Explanation(fsoi, scores_rfi)
ex_rfi.hbarplot()
plt.show()

df_rfi = ex_rfi.fi_means_quantiles()
df_rfi['type'] = 'rfi' """

# df_res = pd.concat([df_pfi, df_cfi, df_rfi]).reset_index()
df_res = pd.concat([df_pfi, df_cfi]).reset_index()
df_res.to_csv(savepath+'df_res.csv')

# print(rf.params)