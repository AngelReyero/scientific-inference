import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
import glob


"""
Plot of double robustness with complex learners from the main text.
"""

parallel=True
p=10
cor=0.6
n_samples=[100, 250, 500, 1000, 2000, 5000]
super_learner=False
y_method = "fixed_poly"
best_model= 'fast_gradBoost'

def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])

def theoretical_curve(y_method, j, correlation,p, beta=[2, 1]):
    """
    Computes the theoretical value for a coordinate `j` based on the specified method.

    Parameters:
    -----------
    y_method : str
        The method used for computation. Can be either 'lin' (linear) or 'nonlin' (nonlinear).
    j : int
        The coordinate index for which the theoretical value is computed.
    correlation : float
        The correlation coefficient.
    p : int
        The dimension of the Toeplitz matrix used in the nonlinear case.
    beta : list, optional
        Coefficients used in the linear case, default is [2, 1].

    Returns:
    --------
    float
        The theoretical value for the given coordinate `j`.
    """
    if y_method == 'lin':
        return beta[j]**2*(1-correlation**2)
    elif y_method == 'nonlin':
        if j >4:
            return 0
        elif j==0:
            return (1-correlation**2)/2
        elif j==1:
            mat=toep(p, correlation)
            sigma_1=mat[1]
            sigma_1=np.delete(sigma_1, 1)
            inv=np.delete(mat, 1, axis=0)
            inv=np.delete(inv, 1, axis=1)
            inv=np.linalg.inv(inv)
            return (1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5
        elif j==2 or j==3:
            mat=toep(p, correlation)
            sigma_1=mat[j]
            sigma_1=np.delete(sigma_1, j)
            inv=np.delete(mat, j, axis=0)
            inv=np.delete(inv, j, axis=1)
            inv=np.linalg.inv(inv)
            return 4*(1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5
    elif y_method == 'fixed_poly':
        if j==0:
            mat=toep(p, correlation)
            sigma_0=mat[0]
            sigma_0=np.delete(sigma_0, 0)
            inv=np.delete(mat, 0, axis=0)
            inv=np.delete(inv, 0, axis=1)
            inv=np.linalg.inv(inv)
            return (correlation-np.dot(np.dot(sigma_0,inv), sigma_0.T))
        elif j==1:
            mat=toep(p, correlation)
            sigma_j=mat[j]
            sigma_j=np.delete(sigma_j, j)
            inv=np.delete(mat, j, axis=0)
            inv=np.delete(inv, j, axis=1)
            inv=np.linalg.inv(inv)
            return (4*(correlation-np.dot(np.dot(sigma_j,inv), sigma_j.T)))
        elif j==4:# var(X²) = 2sigma⁴+4sigma²mu²
            mat=toep(p, correlation)
            sigma_j=mat[j]
            sigma_j=np.delete(sigma_j, j)
            inv=np.delete(mat, j, axis=0)
            inv=np.delete(inv, j, axis=1)
            inv=np.linalg.inv(inv)
            cond_var= (correlation-np.dot(np.dot(sigma_j,inv), sigma_j.T))
            mu = np.zeros(p)
            Sigma = toep(p, cor)  # covariance matrix of X
            rng = np.random.default_rng(0)
            X = rng.multivariate_normal(mu, Sigma, size=(10000))
            X_minus_j = np.delete(X, j, axis=1)
            mn=np.dot(X_minus_j, np.dot(sigma_j,inv))
            return np.mean(2*cond_var**2+4*cond_var*mn**2)
        elif j==7 or j==8:# sigma²*sigma_cond²
            mat=toep(p, correlation)
            sigma_j=mat[j]
            sigma_j=np.delete(sigma_j, j)
            inv=np.delete(mat, j, axis=0)
            inv=np.delete(inv, j, axis=1)
            inv=np.linalg.inv(inv)
            return correlation**2*(cor-np.dot(np.dot(sigma_j,inv), sigma_j.T))
        else:
            return 0


palette = {
    # Sobol-CPI variants → blues
    "Sobol-CPI(1)": "#1f77b4",    # medium blue
    "Sobol-CPI(100)": "#6baed6",  # lighter blue

    # SAGE family → shades of red-purple
    "cSAGE": "#c44e52",           # reddish-pink
    "mSAGE": "#dd1c77",           # deep pink
    "cSAGEvf": "#e377c2",         # pink
    "mSAGEvf": "#bc80bd",         # lavender
    "scSAGEj": "#9e6db5",         # purple

    # LOCO-related → green/cyan
    "LOCO": "#2ca02c",            # green
    "LOCO-W": "#17becf",          # cyan
    "LOCI": "#a1d99b",            # light green

    # Other methods
    "PFI": "#ff7f0e",             # orange
    "CFI": "#d62728",             # red
}

def main():
    if parallel:
        csv_files = glob.glob(f"csv/conv_rates/conv_rates_{y_method}_p{p}_cor{cor}*.csv")
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    elif super_learner:
        df = pd.read_csv(f"csv/conv_rates_{y_method}_p{p}_cor{cor}_super.csv")
    else:
        df = pd.read_csv(f"csv/conv_rates_{y_method}_p{p}_cor{cor}.csv")

    # Normalize
    imp_cols = [col for col in df.columns if col.startswith("imp_V")]
    df[imp_cols] = df[imp_cols].div(df[imp_cols].sum(axis=1), axis=0)



    null_imp = []
    # Iterate through each row of the DataFrame
    indices_to_use = [2, 3, 5, 6, 9]

    for index, row in df.iterrows():
        y_pred = row.filter(like="imp_V").values
        selected_values = [abs(y_pred[i]) for i in indices_to_use if i < len(y_pred)]
        null_imp.append(np.mean(selected_values))

    
    df['null_imp'] = null_imp



    sns.set_style("white")

    # Create figure and GridSpec
    fig = plt.figure(figsize=(16, 6))


    gs = gridspec.GridSpec(1, 3, width_ratios=[1, -0.15, 1], wspace=0.3)  # 0.1 is the spacer

    # Define axes, with a spacer in position 1
    ax0 = fig.add_subplot(gs[0])
    ax_spacer = fig.add_subplot(gs[1])  # this will stay empty
    ax1 = fig.add_subplot(gs[2])

    # Turn off the spacer subplot
    ax_spacer.axis("off")

    axes = [ax0, ax1]
    for axis in axes:
        axis.grid(False)

    # Plot for imp_V0 (subplot 1)
    sns.lineplot(data=df, x='n_samples', y='imp_V0', hue='method', palette=palette, ax=ax0)
    #th_cv_v0 = theoretical_curve(y_method, 0, cor, p, beta=[2, 1])
    #ax0.plot(n_samples, [th_cv_v0 for _ in n_samples], label=r"Theoretical", linestyle='--', linewidth=1, color="black")
    ax0.set_xscale('log')
    ax0.tick_params(axis='x', labelsize=18)
    ax0.tick_params(axis='y', labelsize=18)
    ax0.set_xlabel('')
    ax0.set_ylabel(f'Importance of $X_0$', fontsize=25)
    ax0.legend().remove()


    # Plot for imp_V2 (subplot 0)
    sns.lineplot(data=df, x='n_samples', y='imp_V2', hue='method', palette=palette, ax=ax1)
    #th_cv_v0 = theoretical_curve(y_method, 0, cor, p, beta=[2, 1])
    #ax0.plot(n_samples, [th_cv_v0 for _ in n_samples], label=r"Theoretical", linestyle='--', linewidth=1, color="black")
    ax1.set_xscale('log')
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xlabel('')
    ax1.set_ylabel(f'Importance of $X_2$', fontsize=25)
    ax1.legend(title='Method',  loc='upper right', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.55, 1))#bbox_to_anchor=(1.05, 1),

    # Add common x-axis label
    fig.text(0.5, -0.02, 'Number of samples', ha='center', fontsize=25)


    # Save figure
    if super_learner: 
        plt.savefig(f"figures/convergence_{y_method}_p{p}_cor{cor}_super.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"figures/convergence_{y_method}_p{p}_cor{cor}_model{best_model}.pdf", bbox_inches="tight")



    # Create figure and GridSpec
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, -0.15, 1], wspace=0.3)  # 0.1 is the spacer

    # Define axes, with a spacer in position 1
    ax0 = fig.add_subplot(gs[0])
    ax_spacer = fig.add_subplot(gs[1])  # this will stay empty
    ax1 = fig.add_subplot(gs[2])

    # Turn off the spacer subplot
    ax_spacer.axis("off")

    axes = [ax0, ax1]
    for axis in axes:
        axis.grid(False)
    methods_to_keep = ['Sobol-CPI(1)', 'Sobol-CPI(100)','CFI', 'scSAGEj', 'LOCO', 'LOCO-W']
    df_filtered = df[df['method'].isin(methods_to_keep)]
    # Plot for imp_V0 (subplot 1)
    sns.lineplot(data=df_filtered, x='n_samples', y='imp_V0', hue='method', palette=palette, ax=ax0)
    #th_cv_v0 = theoretical_curve(y_method, 0, cor, p, beta=[2, 1])
    #ax0.plot(n_samples, [th_cv_v0 for _ in n_samples], label=r"Theoretical", linestyle='--', linewidth=1, color="black")
    ax0.set_xscale('log')
    ax0.tick_params(axis='x', labelsize=18)
    ax0.tick_params(axis='y', labelsize=18)
    ax0.set_xlabel('')
    ax0.set_ylabel(f'Importance of $X_0$', fontsize=25)
    ax0.legend().remove()


    # Plot for null importance (subplot 2)
    sns.lineplot(data=df_filtered, x='n_samples', y='null_imp', hue='method', palette=palette, ax=ax1)
    ax1.set_xscale('log')
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xlabel('')
    ax1.set_ylabel('Bias null covariates', fontsize=25)
    ax1.legend(title='Method',  loc='upper right', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.55, 1))#bbox_to_anchor=(1.05, 1),

    # Add common x-axis label
    fig.text(0.5, -0.02, 'Number of samples', ha='center', fontsize=25)


    # Save figure
    if super_learner: 
        plt.savefig(f"figures/convergence_reduced_{y_method}_p{p}_cor{cor}_super.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"figures/convergence_reduced_{y_method}_p{p}_cor{cor}_model{best_model}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()