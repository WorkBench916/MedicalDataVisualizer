import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
df['overweight'] = np.where(df['weight'] / ((df['height'] / 100) ** 2) >= 25, 1, 0 )

# 3
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(frame = df,
                 id_vars = ['cardio'],
                 value_vars = ['cholesterol','gluc','smoke', 'alco', 'active', 'overweight'],
                 var_name = 'variable',
                 value_name = 'value')


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7

    # 8
    fig = sns.catplot(data = df_cat, 
            x = 'variable', 
            y = 'total', 
            hue = 'value',
            col = 'cardio',
            kind = 'bar').fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    dialostic_check = df['ap_lo'] <= df['ap_hi']
    height_lower = df['height'] >= df['height'].quantile(0.025)
    height_upper = df['height'] <= df['height'].quantile(0.975)
    weight_lower = df['weight'] >= df['weight'].quantile(0.025)
    weight_upper = df['weight'] <= df['weight'].quantile(0.975)

    df_heat = df[
        (dialostic_check) &
        (height_lower) &
        (height_upper) &
        (weight_lower) &
        (weight_upper)
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(10,8))

    # 15
    sns.heatmap(data = corr, mask = mask, ax = ax, cmap='coolwarm', annot = True, fmt ='.1f')


    # 16
    fig.savefig('heatmap.png')
    return fig
