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
p=10
y_method = "bike"
best_model='fast_gradBoost'#'rf'#'fast_gradBoost'#'rf'# #
csv_files = glob.glob(f"csv/conv_rates/bike_{y_method}_model{best_model}_seed*.csv")
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



# Normalize
imp_cols = [col for col in df.columns if col.startswith("imp_V")]
df[imp_cols] = df[imp_cols].div(df[imp_cols].sum(axis=1), axis=0)
# Plot boxplot
methods_to_keep = ['Sobol-CPI(1)', 'Sobol-CPI(100)','CFI', 'scSAGEj', 'LOCO']
df_filtered = df[df['method'].isin(methods_to_keep)]
plt.figure(figsize=(16, 8))
df_long = pd.melt(df_filtered, id_vars='method', 
                  value_vars=[f'imp_V{i}' for i in range(0, p)],
                  var_name='Variable', value_name='Importance')

# Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

# Plot

plt.figure(figsize=(12, 6))
var_mapping = {
    'V0': 'season',
    'V1': 'yr',
    'V2': 'mnth',
    'V3': 'holiday',
    'V4': 'weekday',
    'V5': 'workingday',
    'V6': 'weathersit',
    'V7': 'temp',
    'V8': 'hum',
    'V9': 'windspeed',
    'V10': 'days_since_2011'
}

# Replace the values in 'Variable'
df_long['Variable'] = df_long['Variable'].map(var_mapping)
ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)
plt.ylim(-1, 1)


ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

plt.xticks(rotation=45)
mean_r2 = df['r2'].mean()
plt.title(fr"Feature importance for bike dataset, $R^2 = {mean_r2:.2f}$", fontsize=25)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.ylabel('Importance', fontsize=20)
plt.xlabel('Variables', fontsize=20)
plt.legend(title='Method',  loc='upper left', fontsize=16, title_fontsize=18, ncol=2)#bbox_to_anchor=(1.05, 1),
plt.tight_layout()
plt.savefig(f"figures/{y_method}_model{best_model}.pdf", bbox_inches="tight")




# Long boxplot:

# Plot boxplot
plt.figure(figsize=(16, 8))
df_long = pd.melt(df, id_vars='method', 
                  value_vars=[f'imp_V{i}' for i in range(0, p)],
                  var_name='Variable', value_name='Importance')

# Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

# Plot

plt.figure(figsize=(12, 6))
var_mapping = {
    'V0': 'season',
    'V1': 'yr',
    'V2': 'mnth',
    'V3': 'holiday',
    'V4': 'weekday',
    'V5': 'workingday',
    'V6': 'weathersit',
    'V7': 'temp',
    'V8': 'hum',
    'V9': 'windspeed',
    'V10': 'days_since_2011'
}

# Replace the values in 'Variable'
df_long['Variable'] = df_long['Variable'].map(var_mapping)
ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)


ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

plt.xticks(rotation=45)
mean_r2 = df['r2'].mean()
plt.title(fr"Feature importance for bike dataset, $R^2 = {mean_r2:.2f}$", fontsize=25)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.ylim(-1, 1)
plt.ylabel('Importance', fontsize=20)
plt.xlabel('Variables', fontsize=20)
plt.legend(title='Method',  loc='upper left', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.05, 1))#bbox_to_anchor=(1.05, 1),
plt.tight_layout()
plt.savefig(f"figures/{y_method}_complete_model{best_model}.pdf", bbox_inches="tight")





# Plot boxplot
plt.figure(figsize=(16, 8))
df_long = pd.melt(df, id_vars='method', 
                  value_vars=[f'imp_V{i}' for i in range(1, 2)],
                  var_name='Variable', value_name='Importance')

# Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

# Plot

plt.figure(figsize=(12, 6))
var_mapping = {
    'V0': 'season',
    'V1': 'yr',
    'V2': 'mnth',
    'V3': 'holiday',
    'V4': 'weekday',
    'V5': 'workingday',
    'V6': 'weathersit',
    'V7': 'temp',
    'V8': 'hum',
    'V9': 'windspeed',
    'V10': 'days_since_2011'
}

# Replace the values in 'Variable'
df_long['Variable'] = df_long['Variable'].map(var_mapping)
ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)


ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

plt.xticks(rotation=45)
mean_r2 = df['r2'].mean()
plt.title(fr"Feature importance for bike dataset, $R^2 = {mean_r2:.2f}$", fontsize=25)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.ylim(-1, 1)
plt.ylabel('Importance', fontsize=20)
plt.xlabel('Variables', fontsize=20)
plt.legend(title='Method',  loc='upper left', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.05, 1))#bbox_to_anchor=(1.05, 1),
plt.tight_layout()
plt.savefig(f"figures/{y_method}_reduced_model{best_model}.pdf", bbox_inches="tight")







