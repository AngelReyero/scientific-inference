import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf

import rfi.examples.chains as chains
from rfi.explainers.explainer import Explainer
from rfi.explanation.explanation import Explanation
from rfi.explanation.decomposition import DecompositionExplanation
from rfi.samplers.simple import SimpleSampler
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
from rfi.decorrelators.naive import NaiveDecorrelator

import logging
import math

logging.basicConfig(level=logging.INFO)

# savepath = 'C:/Users/ra59qih/sciebo/LMU/Forschung/Feature_importance/Python/'
savepath = ''
# rpath = 'C:/Users/ra59qih/sciebo/LMU/Forschung/Feature_importance/R/'

# Example 1

# reg_lin = linear_model.LinearRegression()


# datasets to use
data = pd.read_csv(savepath + 'interaction.csv')

data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'y']]
ntrain = int(0.7 * data.shape[0])

xcolumns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
ycolumn = ['y']
df_train, df_test = data.iloc[0:ntrain,], data.iloc[ntrain:,]
X_train, y_train = df_train[xcolumns], df_train[ycolumn]
X_test, y_test = df_test[xcolumns], df_test[ycolumn]

# fit models
# reg_lin.fit(X_train, y_train)
# reg_lin.coef_[0, 1] = reg_lin.coef_[0, 1] + 0.05
# reg_lin.coef_[0, 5] = reg_lin.coef_[0, 5] + reg_lin.coef_[0, 6]
# reg_lin.coef_[0, 6] = 0

reg_lin = smf.ols(formula='y ~ x1 + np.square(x1) + x2 + np.square(x2) + x3 + np.square(x3) + x4 + np.square(x4) + x5 + np.square(x5) + x6 + np.square(x6) + x7 + np.square(x7) + x1:x2 + x1:x3 + x1:x4 + x1:x5 + x1:x6 + x1:x7 + x2:x3 + x2:x4 + x2:x5 + x2:x6 + x2:x7 + x3:x4 + x3:x5 + x3:x6 + x3:x7 + x4:x5 + x4:x6 + x4:x7 + x5:x6 + x5:x7 + x6:x7', data=df_train).fit()


scoring = [mean_squared_error, r2_score]
names = ['MSE', 'r2_score']
models = [reg_lin]
m_names = ['LinearRegression']

for kk in range(len(models)):
    model = models[kk]
    print('Model: {}'.format(m_names[kk]))
    for jj in np.arange(len(names)):
        print('{}: {}'.format(names[jj],
                              scoring[jj](y_test, model.predict(X_test))))

# explain model

sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)
fsoi = X_train.columns
ordering = [tuple(fsoi)]

wrk = Explainer(reg_lin.predict, fsoi, X_train,
                loss=mean_squared_error, sampler=sampler,
                decorrelator=decorrelator)

ex_sage = wrk.ais_via_contextfunc(fsoi, X_test, y_test, context='empty', marginalize=True)
ex_sage.hbarplot()
plt.show()

df_sage = ex_sage.fi_means_quantiles()
df_sage['type'] = 'conditional v(j)'

ex_sage2 = wrk.ais_via_contextfunc(fsoi, X_test, y_test, context='remainder', marginalize=True)
ex_sage2.hbarplot()
plt.show()

df_sage2 = ex_sage2.fi_means_quantiles()
df_sage2['type'] = 'conditional v(-j u j) - v(-j)'

ex_sage_m = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='empty', marginalize=True)
ex_sage_m.hbarplot()
plt.show()

df_sage_m = ex_sage_m.fi_means_quantiles()
df_sage_m['type'] = 'marginal v(j)'

ex_sage_m2 = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='remainder', marginalize=True)
ex_sage_m2.hbarplot()
plt.show()

df_sage_m2 = ex_sage_m2.fi_means_quantiles()
df_sage_m2['type'] = 'marginal v(-j u j) - v(-j)'


# saving overall result

df_res2 = pd.concat([df_sage, df_sage2, df_sage_m, df_sage_m2]).reset_index()
df_res2.to_csv(savepath+'df2_res2.csv')

print(reg_lin.params)

## SAGE

df = data

ex_msage, orderings = wrk.sage(X_test, y_test, ordering, method='direct')
ex_msage.ex_name = 'msage'
ex_msage.to_csv(savepath=savepath, filename='ex_msage.csv')
ex_msage = Explanation.from_csv(savepath+'ex_msage.csv')

ex_msage.hbarplot()
plt.show()

df_msage = ex_msage.fi_means_quantiles()
df_msage['type'] = 'mSAGE'

ex_csage, orderings = wrk.sage(X_test, y_test, ordering, method='associative')
ex_csage.ex_name = 'csage'
ex_csage.to_csv(savepath=savepath, filename='ex_csage.csv')
ex_csage = Explanation.from_csv(savepath+'ex_csage.csv')

ex_csage.hbarplot()
plt.show()

df_csage = ex_csage.fi_means_quantiles()
df_csage['type'] = 'cSAGE'

df_interactions_res = pd.concat([df_msage, df_csage]).reset_index()
df_interactions_res.to_csv(savepath+'df2_res_SAGE.csv')