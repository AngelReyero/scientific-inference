import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from fippy.explainers import Explainer
from fippy.explanation import Explanation
from fippy.samplers import GaussianSampler

import logging

import statsmodels.api as sm
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO)

# savepath = 'C:/Users/ra59qih/sciebo/LMU/Forschung/Feature_importance/Python/'
savepath = ''
# rpath = 'C:/Users/ra59qih/sciebo/LMU/Forschung/Feature_importance/R/'
# savepath = '~/university/postdoc/research/fi_inference/code/paper_2022_feature_importance_guide/Simulation/Python/'
# Example 1

# reg_lin = linear_model.LinearRegression()
rf = RandomForestRegressor()


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
# reg_lin= smf.ols(formula='y ~ x1 + x2 + x3 + np.square(x3) + x4 + np.square(x4) + x5 + np.square(x5) + x3:x4 + x3:x5 + x4:x5', data=df_train).fit()
rf.fit(X_train, y_train)
#reg_lin.params[9] = 0
#reg_lin.params[10] = 0.2960963746

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

sampler = GaussianSampler(X_train)
wrk = Explainer(rf.predict, X_train, loss=mean_squared_error, sampler=sampler)

fsoi = list(X_train.columns)
ex_sage = wrk.csagevfs(X_test, y_test, C='empty', nr_resample_marginalize=100)
ex_sage = wrk.ais_via_contextfunc(fsoi, X_test, y_test, context='empty', marginalize=True)
# ex_sage.hbarplot()
ex_sage.hbarplot()
plt.show()

df_sage = ex_sage.fi_means_quantiles()
df_sage['type'] = 'conditional v(j)'

ex_sage2 = wrk.csagevfs(X_test, y_test, C='remainder', nr_resample_marginalize=100)
# ex_sage2 = wrk.ais_via_contextfunc(fsoi, X_test, y_test, context='remainder', marginalize=True)
ex_sage2.hbarplot()
plt.show()

df_sage2 = ex_sage2.fi_means_quantiles()
df_sage2['type'] = 'conditional v(-j u j) - v(-j)'

ex_sage_m = wrk.msagevfs(X_test, y_test, C='empty', nr_resample_marginalize=100)
# ex_sage_m = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='empty', marginalize=True)
ex_sage_m.hbarplot()
plt.show()

df_sage_m = ex_sage_m.fi_means_quantiles()
df_sage_m['type'] = 'marginal v(j)'

ex_sage_m2 = wrk.msagevfs(X_test, y_test, C='remainder', nr_resample_marginalize=100)
# ex_sage_m2 = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='remainder', marginalize=True)
ex_sage_m2.hbarplot()
plt.show()

df_sage_m2 = ex_sage_m2.fi_means_quantiles()
df_sage_m2['type'] = 'marginal v(-j u j) - v(-j)'


# saving overall result

df_res2 = pd.concat([df_sage, df_sage2, df_sage_m, df_sage_m2]).reset_index()
df_res2.to_csv(savepath+'df_res2.csv')

# print(rf.params)

## SAGE

# df = data

#mSAGE
ex_msage, orderings = wrk.msage(X_test, y_test, nr_resample_marginalize=100)
ex_msage.ex_name = 'msage'
# ex_msage.to_csv(savepath=savepath, filename='ex_msage.csv')
# ex_msage = Explanation.from_csv(savepath+'ex_msage.csv')

ex_msage.hbarplot()
plt.show()

df_msage = ex_msage.fi_means_quantiles()
df_msage['type'] = 'mSAGE'

# cSAGE 
ex_csage, orderings = wrk.csage(X_test, y_test, nr_resample_marginalize=100)
ex_csage.ex_name = 'csage'
#ex_csage.to_csv(savepath=savepath, filename='ex_csage.csv')
#ex_csage = Explanation.from_csv(savepath+'ex_csage.csv')

ex_csage.hbarplot()
plt.show()

df_csage = ex_csage.fi_means_quantiles()
df_csage['type'] = 'cSAGE'

df_interactions_res = pd.concat([df_msage, df_csage]).reset_index()
df_interactions_res.to_csv(savepath+'df_res_SAGE.csv')