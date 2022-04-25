from utils import *

folder2save = "./Data/modeling"
data = pd.read_csv("Data/TimeSeries_v5_all_data.csv")

# %% process data
(features_time_inv, features_time_var) = process_data(data.copy())

# %% generate dataset
generate_test_set4transfer_learning(720, folder2save, features_time_inv, features_time_var)

generate_test_set4transfer_learning(1080, folder2save, features_time_inv, features_time_var)
