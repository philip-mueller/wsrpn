
import os
from src.settings import PROJECT_DIR
import pandas as pd
from scipy.stats import ttest_ind


metric_names = ['RoDeO/total', 'RoDeO/classification', 'RoDeO/localization', 'RoDeO/shape_matching', 'mAP_thres/mAP@0.3', 'mAP_thres/mAP@0.5', 'acc_thres/acc@0.3', 'acc_thres/acc@0.5'] 

wsrpn_path = os.path.join(PROJECT_DIR, 'results', 'bootstrapping', 'wsrpn.csv')
wsrpn_df = pd.read_csv(wsrpn_path)[metric_names]
wsrpn_df[['RoDeO/total', 'RoDeO/classification', 'RoDeO/localization', 'RoDeO/shape_matching']] *= 100


chexnet_noisyor_path = os.path.join(PROJECT_DIR, 'results', 'bootstrapping', 'chexnet_noisyor.csv')
chexnet_noisyor_df = pd.read_csv(chexnet_noisyor_path)[metric_names]
chexnet_noisyor_df[['RoDeO/total', 'RoDeO/classification', 'RoDeO/localization', 'RoDeO/shape_matching']] *= 100

for metric in metric_names:
    t_statistic, p_value = ttest_ind(wsrpn_df[metric], chexnet_noisyor_df[metric], equal_var=False, alternative='greater')
    print(f'{metric}: t_statistic={t_statistic}, p_value={p_value:.10f}')

