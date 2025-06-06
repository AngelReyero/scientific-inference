import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob



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
best_model= 'rf'#'fast_gradBoost'
p=10
cor=0.6
n=5000
csv_files = glob.glob(f"csv/conv_rates/conv_rates_{y_method}_p{p}_cor{cor}_model{best_model}_seed*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

palette = {
    "Sobol-CPI(1)": "#1f77b4",    # blue
    "Sobol-CPI(100)": "#ff7f0e",  # orange
    "PFI": "#d2691e",             # chocolate (dark orange, NOT purple)
    "CFI": "#2ca02c",             # green
    "cSAGEvf": "#d62728",         # red
    "scSAGEj": "#9467bd",         # medium purple
    "mSAGEvf": "#8c564b",         # brown
    "cSAGE": "#e377c2",           # pink
    "mSAGE": "#7f7f7f",           # gray
    "LOCO": "#bcbd22",            # olive green
    "LOCO-W": "#17becf",          # cyan
    "LOCI": "#aec7e8",            # light blue
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
df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)
green_vars = {'V0', 'V1', 'V4', 'V7', 'V8'}
for label in ax.get_xticklabels():
    var_name = label.get_text()
    label.set_color('green' if var_name in green_vars else 'red')
plt.xticks(rotation=45)
mean_r2 = filtered_df['r2'].mean()
plt.title(fr"Feature importance for $y = X_0 + 2X_1 - X_4^2 + X_7X_8$, $R^2 = {mean_r2:.2f}$")
plt.legend(title='Method',  loc='upper right')#bbox_to_anchor=(1.05, 1),
plt.tight_layout()
plt.savefig(f"figures/conv_rates_{y_method}_p{p}_cor{cor}_model{best_model}.pdf", bbox_inches="tight")
