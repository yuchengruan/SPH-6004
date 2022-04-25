# %%
from utils import *

folder2save = "./Data/modeling_v2"
os.makedirs(folder2save, exist_ok=True)
data = pd.read_csv("Data/TimeSeries_v5_all_data.csv")

# %% 1. pre-process the time-invariant and time-variant features
(features_time_inv, features_time_var) = process_data(data.copy())

# %% 2. generate datasets in different time intervals
features_time_inv_with_aki = features_time_inv[features_time_inv["aki_onset_minute"].notnull()]
features_time_inv_without_aki = features_time_inv[features_time_inv["aki_onset_minute"].isnull()]

generate_dataset(360, folder2save, features_time_inv_with_aki, features_time_inv_without_aki,
                 features_time_var)
generate_dataset(720, folder2save, features_time_inv_with_aki, features_time_inv_without_aki,
                 features_time_var)
generate_dataset(1080, folder2save, features_time_inv_with_aki, features_time_inv_without_aki,
                 features_time_var)
