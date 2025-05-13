import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
bmi_df = df['weight'] / (df['height'] / 100) * (df['height'] / 100)

df['overweight'] = np.where(df['weight'] / ((df['height']/100)**2) > 25, 1, 0)

# 3
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'smoke', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6
    df_cat = df_cat.groupby('cardio').value_counts().reset_index(name='total')
    # 7
    vars = sorted(['cholesterol', 'smoke', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        col='cardio',
        kind='bar',
        order=vars,
        hue='value'
    )

    # 8
    fig.set(xlabel='variable', ylabel='total')
    fig = fig.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) & 
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 12
    corr = df_heat.corr(numeric_only=True)

    # 13
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # 14
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15
    ax = sns.heatmap(
        corr,
        mask=mask,
        vmax=0.4,
        fmt ='.1f',
        annot=True
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
