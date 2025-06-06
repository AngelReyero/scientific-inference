from loco import LOCO
from loci import LOCI
from fippy.explainers import Explainer
from fippy.samplers import GaussianSampler
import numpy as np
import vimpy
from sobol_CPI import Sobol_CPI
import pandas as pd
from utils import best_mod, GenToysDataset
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convergence rates")
    parser.add_argument("--seeds", type=int, nargs="+", help="List of seeds")
    return parser.parse_args()


def main(args):
    

    for s in args.seeds:
        y_method = "fixed_poly"
        p=10
        cor=0.6
        n_samples=[100, 250, 500, 1000, 2000, 5000]
        cor_meth='toep'
        sparsity=0.25
        super_learner=False
        n_cal=100
        n_jobs=10

        best_model= 'rf' #'gradBoost' #  'fast_gradBoost'
        dict_model=None

        rng = np.random.RandomState(s)

        importance_score=np.zeros((12, len(n_samples), p))# 12 because there are 12 methods
        true_importance=np.zeros((len(n_samples), p))
        r2_sc = np.zeros(len(n_samples))
        mse = np.zeros(len(n_samples))
        print("Experiment: "+str(s))
        for (i,n) in enumerate(n_samples):
            print("With n="+str(n))
            X, y, true_imp = GenToysDataset(n=n, d=p, cor=cor_meth, y_method=y_method, mu=None, rho_toep=cor,  sparsity=sparsity, seed=s)
            true_importance[i]=true_imp
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=s)
            feature_names = [f"X{i}" for i in range(X_train.shape[1])]
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            y_test_df = pd.DataFrame(y_test)
            y_train_df = pd.DataFrame(y_train)            
            model=best_mod(X_train_df, y_train, seed=s, regressor=best_model, dict_reg=dict_model, super_learner=super_learner)

            scoring = [mean_squared_error, r2_score]
            names = ['MSE', 'r2_score']

            for jj in np.arange(len(names)):
                print('{}: {}'.format(names[jj], scoring[jj](y_test, model.predict(X_test))))
            r2_sc[i] = r2_score(y_test, model.predict(X_test))
            mse[i] = mean_squared_error(y_test, model.predict(X_test))

            sobol_cpi= Sobol_CPI(
                estimator=model,
                imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5),
                n_permutations=1,
                random_state=s,
                n_jobs=n_jobs)
            sobol_cpi.fit(X_train, y_train)

            cpi_importance = sobol_cpi.score(X_test, y_test, n_cal=1)
            importance_score[0,i]= cpi_importance["importance"].reshape((p,))

            sobol_importance = sobol_cpi.score(X_test, y_test, n_cal=n_cal)
            importance_score[1,i]= sobol_importance["importance"].reshape((p,))
            

            sampler = GaussianSampler(X_train_df)
            wrk = Explainer(model.predict, X_train_df, loss=mean_squared_error, sampler=sampler)

            # PFI
            pfi = wrk.pfi(X_test_df, y_test_df)
            pfi_scores = pfi.fi_means_stds()
            importance_score[2,i]= np.array([
                pfi_scores[k] for k in pfi_scores.index if k != 'std'
            ])
            # CFI
            cfi = wrk.cfi(X_test_df, y_test_df)
            cfi_scores = cfi.fi_means_stds()
            importance_score[3,i]= np.array([
                cfi_scores[k] for k in cfi_scores.index if k != 'std'
            ])
            # cSAGEvf
            cSAGEvf = wrk.csagevfs(X_test_df, y_test_df, C='empty', nr_resample_marginalize=50)
            cSAGEvf_scores = cSAGEvf.fi_means_stds()
            importance_score[4,i]= np.array([
                cSAGEvf_scores[k] for k in cSAGEvf_scores.index if k != 'std'
            ])
            # scSAGEvfj
            scSAGEvfj = wrk.csagevfs(X_test_df, y_test_df, C='remainder', nr_resample_marginalize=50)
            scSAGEvfj_scores = scSAGEvfj.fi_means_stds()
            importance_score[5,i]= np.array([
                scSAGEvfj_scores[k] for k in scSAGEvfj_scores.index if k != 'std'
            ])
            # mSAGEvf
            mSAGEvf = wrk.msagevfs(X_test_df, y_test_df, C='empty', nr_resample_marginalize=50)
            mSAGEvf_scores = mSAGEvf.fi_means_stds()
            importance_score[6,i]= np.array([
                mSAGEvf_scores[k] for k in mSAGEvf_scores.index if k != 'std'
            ])
                
            # cSAGE
            cSAGE, ordering = wrk.csage(X_test_df, y_test_df, nr_resample_marginalize=50)
            cSAGE_scores = cSAGE.fi_means_stds()
            importance_score[7,i]= np.array([
                cSAGE_scores[k] for k in cSAGE_scores.index if k != 'std'
            ])   

            # mSAGE
            mSAGE, ordering = wrk.msage(X_test_df, y_test_df, nr_resample_marginalize=50)
            mSAGE_scores = mSAGE.fi_means_stds()
            importance_score[8,i]= np.array([
                mSAGE_scores[k] for k in mSAGE_scores.index if k != 'std'
            ]) 

        
            #LOCO Williamson
            ntrees = np.arange(100, 500, 100)
            lr = np.arange(.01, .1, .05)
            param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
            ## set up cv objects
            cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 5, n_jobs=n_jobs)
            for j in range(p):
                print("covariate: "+str(j))
                vimp = vimpy.vim(y = y, x = X, s = j, pred_func = cv_full, measure_type = "r_squared")
                vimp.get_point_est()
                vimp.get_influence_function()
                vimp.get_se()
                vimp.get_ci()
                vimp.hypothesis_test(alpha = 0.05, delta = 0)
                importance_score[10,i,j]+=vimp.vimp_*np.var(y)

            loco = LOCO(
                estimator=model,
                random_state=s,
                loss = mean_squared_error,
                n_jobs=n_jobs,
            )
            loco.fit(X_train, y_train)
            loco_importance = loco.score(X_test, y_test)
            importance_score[9,i]= loco_importance["importance"].reshape((p,))

            loci = LOCI(
                estimator=model,
                random_state=s,
                loss=mean_squared_error,
                n_jobs=n_jobs,
            )

            loci.fit(X_train_df, y_train)
            loci_importance = loci.score(X_test_df, y_test)
            importance_score[11,i]= np.array(list(loci_importance.values()))

        #Save the results
        f_res={}
        f_res = pd.DataFrame(f_res)
        for i in range(12):
            for j in range(len(n_samples)):
                f_res1={}
                if i==0:
                    f_res1["method"] = ["Sobol-CPI(1)"]
                elif i==1:
                    f_res1["method"]=["Sobol-CPI(100)"]
                elif i==2: 
                    f_res1["method"]=["PFI"]
                elif i==3:
                    f_res1["method"]=["CFI"]
                elif i==4:
                    f_res1["method"] = ["cSAGEvf"]
                elif i==5:
                    f_res1["method"]=["scSAGEj"]
                elif i==6: 
                    f_res1["method"]=["mSAGEvf"]
                elif i==7:
                    f_res1["method"]=["cSAGE"]
                elif i==8:
                    f_res1["method"] = ["mSAGE"]
                elif i==9:
                    f_res1["method"]=["LOCO"]
                elif i==10: 
                    f_res1["method"]=["LOCO-W"]
                elif i==11:
                    f_res1["method"]=["LOCI"]        
                f_res1["n_samples"]=n_samples[j]
                f_res1["mse"]=mse[j]
                f_res1["r2"]=r2_sc[j]
                for k in range(p):
                    f_res1["imp_V"+str(k)]=importance_score[i, j, k]
                    f_res1["tr_V"+str(k)] =true_importance[j, k]
                f_res1=pd.DataFrame(f_res1)
                f_res=pd.concat([f_res, f_res1], ignore_index=True)
        if super_learner:
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../results/csv/conv_rates/conv_rates_{y_method}_p{p}_cor{cor}_super_seed{s}.csv"))
            f_res.to_csv(
            csv_path,
            index=False,
            ) 
        else:
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../results/csv/conv_rates/conv_rates_{y_method}_p{p}_cor{cor}_model{best_model}_seed{s}.csv"))
            f_res.to_csv(
            csv_path,
            index=False,
            ) 



# This is the main entry point of the script. It will be executed when the script is 
# run directly, i.e. `python python_script.py --seeds 1 2 3`.
if __name__ == "__main__":
    args = parse_args()
    main(args)


