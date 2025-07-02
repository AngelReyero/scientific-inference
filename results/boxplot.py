import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
from plt_conv_rates import theoretical_curve



# Dummy data setup (replace with your actual df)
methods = [
    "Sobol-CPI(1)",
    "Sobol-CPI(100)",
    "PFI",
    "CFI",
    "cSAGEvf",
    "scSAGEj",
    "mSAGEvf",
    "cSAGE",
    "mSAGE",
    "LOCO",
    "LOCO-W",
    "LOCI"
]

features = [f'X{i}' for i in range(10)]

y_method = "fixed_poly"
best_model='rf'#'fast_gradBoost'#'rf'# #
p=10
cor=0.6
n=5000
csv_files = glob.glob(f"csv/conv_rates/conv_rates_{y_method}_p{p}_cor{cor}_model{best_model}_seed*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

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



filtered_df = df[df['n_samples'] == n].copy()
# Normalize
imp_cols = [col for col in filtered_df.columns if col.startswith("imp_V")]
filtered_df[imp_cols] = filtered_df[imp_cols].div(filtered_df[imp_cols].sum(axis=1), axis=0)
# Plot boxplot
plt.figure(figsize=(16, 8))
df_long = pd.melt(filtered_df, id_vars='method', 
                  value_vars=[f'imp_V{i}' for i in range(0, p)],
                  var_name='Variable', value_name='Importance')

# Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
df_long['Variable'] = df_long['Variable'].str.replace('imp_V', 'X')

# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)
green_vars = {'X0', 'X1', 'X4', 'X7', 'X8'}
for label in ax.get_xticklabels():
    var_name = label.get_text()
    label.set_color('green' if var_name in green_vars else 'red')


#True values
#true_values = {}
#for j in range(p):
#   true_values['V'+str(j)]=theoretical_curve(y_method, j, cor,p)

#total = sum(true_values.values())
#for k in true_values:
#    true_values[k] /= total
# Ensure order matches x-tick labels
#x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
#true_y = [true_values[label] for label in x_labels]
#x = range(len(x_labels))
# Plot dotted step line
#for i in range(len(x)):
#    ax.hlines(y=true_y[i], xmin=x[i] - 0.4, xmax=x[i] + 0.4,
#             colors='black', linestyles=':', linewidth=2, label=None)
# Add one dummy for legend (only once)
#from matplotlib.lines import Line2D
#step_line = Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='True importance')
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles + [step_line], labels + ['True importance'])


ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

plt.xticks(rotation=45)
mean_r2 = filtered_df['r2'].mean()
plt.title(fr"Feature importance for $y = X_0 + 2X_1 - X_4^2 + X_7X_8$, $R^2 = {mean_r2:.2f}$", fontsize=25)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.ylabel('Importance', fontsize=20)
plt.xlabel('Variables', fontsize=20)
if best_model=='fast_gradBoost':
    plt.legend(title='Method',  loc='upper right', fontsize=16, title_fontsize=18, ncol=2)#bbox_to_anchor=(1.05, 1),
else: 
    plt.legend(title='Method',  loc='upper right', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.35, 1))#bbox_to_anchor=(1.05, 1),

plt.tight_layout()
plt.savefig(f"figures/boxplot_{y_method}_p{p}_cor{cor}_model{best_model}.pdf", bbox_inches="tight")





# Reduced boxplot
df_long = pd.melt(filtered_df, id_vars='method', 
                  value_vars=[f'imp_V{i}' for i in [0, 2]],
                  var_name='Variable', value_name='Importance')

# Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
df_long['Variable'] = df_long['Variable'].str.replace('imp_V', 'X')

# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)
green_vars = {'X0', 'X1', 'X4', 'X7', 'X8'}
for label in ax.get_xticklabels():
    var_name = label.get_text()
    label.set_color('green' if var_name in green_vars else 'red')



ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

plt.xticks(rotation=45)
mean_r2 = filtered_df['r2'].mean()
plt.title(fr"Feature importance for $y = X_0 + 2X_1 - X_4^2 + X_7X_8$, $R^2 = {mean_r2:.2f}$", fontsize=25)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.ylabel('Importance', fontsize=20)
plt.xlabel('Variables', fontsize=20)
plt.legend(title='Method',  loc='upper right', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.4, 1))#bbox_to_anchor=(1.05, 1),
plt.tight_layout()
plt.savefig(f"figures/reduced_boxplot_{y_method}_p{p}_cor{cor}_model{best_model}.pdf", bbox_inches="tight")

