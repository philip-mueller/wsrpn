
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import seaborn as sns

from src.settings import PROJECT_DIR

class_names: List[str] = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
metric_names = [f'RoDeO/{cls_name}/total' for cls_name in class_names]

wsrpn_path = os.path.join(PROJECT_DIR, 'results', 'bootstrapping', 'wsrpn.csv')
wsrpn_df = pd.read_csv(wsrpn_path)[metric_names] * 100
wsrpn_df = wsrpn_df.rename(columns={old_name: new_name for old_name, new_name in zip(metric_names, class_names)})
wsrpn_df['Model'] = 'WSRPN (ours)'
# flatten class names into single "class_name" column

chexnet_noisyor_path = os.path.join(PROJECT_DIR, 'results', 'bootstrapping', 'chexnet_noisyor.csv')
chexnet_noisyor_df = pd.read_csv(chexnet_noisyor_path)[metric_names] * 100
chexnet_noisyor_df = chexnet_noisyor_df.rename(columns={old_name: new_name for old_name, new_name in zip(metric_names, class_names)})
chexnet_noisyor_df['Model'] = 'CheXNet w/ noisy-or'

df = pd.concat([wsrpn_df, chexnet_noisyor_df])
df = df.melt(id_vars=['Model'], var_name='Pathology', value_name='RoDeO total [%]')

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=800)
sns.violinplot(ax=ax, data=df, x="Pathology", y="RoDeO total [%]", hue="Model", split=True, inner="quart", linewidth=0.5, scale="width")
#sns.boxplot(ax=ax, data=df, x="Pathology", y="RoDeO total [%]", hue="Model", linewidth=1, width=0.8)

# rotate x-axis labels
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
# # Insert vertical lines between the violin pairs
for i in range(1, len(metric_names)):
    ax.axvline(i - 0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

# set font size of legend and ticks and axis labels
ax.legend(fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
# no x label
ax.set_xlabel('')
ax.set_ylabel('RoDeO total [%]', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, 'results', 'bootstrapping', 'violin.pdf'))