import os
from datetime import datetime

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MIMIC4Classification(torch.utils.data.Dataset):

    def __init__(self, data_time_inv_path, data_time_var_path):
        self.feat_time_inv = pd.read_csv(data_time_inv_path)
        self.feat_time_var = pd.read_csv(data_time_var_path)

    def __len__(self):
        return len(self.feat_time_inv)

    def __getitem__(self, idx):
        stay_id = self.feat_time_inv["stay_id"][idx]
        label = self.feat_time_inv["output"][idx]
        time_inv_idx = self.feat_time_inv.iloc[idx, 1:-2]
        time_var_idx = self.feat_time_var[self.feat_time_var["stay_id"] == stay_id].sort_values(
            by=["event_time_minute"]).iloc[:, 1:-1]

        return time_inv_idx.astype("float32").values, time_var_idx.astype("float32").values, label


def eval_classification(rnn_fusion, data_loader, device, criterion=None, report_loss=True, report_metrics=False):
    rnn_fusion.eval()

    Y_pred = []
    Y_true = []
    with torch.no_grad():
        for (time_inv, time_var, label) in data_loader:
            time_inv = time_inv.to(device)
            time_var = time_var.to(device)
            label = label.long().to(device)

            # Forward pass
            y_pred = rnn_fusion(time_inv, time_var)

            Y_true.append(label.cpu())
            Y_pred.append(y_pred[0].cpu())

    Y_true = torch.stack(Y_true, dim=1).squeeze()
    Y_pred = torch.stack(Y_pred)

    if report_loss:
        # Compute loss
        loss = criterion(Y_pred, Y_true).item()
    else:
        loss = None

    # compute metrics
    if report_metrics:
        Y_pred = torch.nn.functional.softmax(Y_pred, dim=1)
        auc_roc = roc_auc_score(Y_true.numpy(), Y_pred.numpy()[:, 1])
        f1 = f1_score(Y_true.numpy(), torch.argmax(Y_pred, dim=1).numpy())
        acc = accuracy_score(Y_true.numpy(), torch.argmax(Y_pred, dim=1).numpy())
    else:
        auc_roc = f1 = acc = None

    return loss, (auc_roc, f1, acc)


def generate_dataset(time_point, folder2save, features_time_inv_with_aki,
                     features_time_inv_without_aki, features_time_var):
    print(f"Dataset [{time_point}, {time_point + 360}]:")
    dataset_path = os.path.join(folder2save, "_".join(["Dataset", str(time_point), str(time_point + 360)]))
    os.makedirs(dataset_path, exist_ok=True)

    # combine patients of aki onset time between [start_time, end_time] with patients without aki for time-invariant features
    features_time_inv_with_aki_extracted = features_time_inv_with_aki[
        ((features_time_inv_with_aki["aki_onset_minute"] > time_point) &
         (features_time_inv_with_aki["aki_onset_minute"] <= time_point + 360))]
    features_time_inv_dataset = pd.concat([features_time_inv_with_aki_extracted,
                                           features_time_inv_without_aki], axis=0)

    # extract time-variant features before start_time and remove the information afterwards
    features_time_var_dataset = features_time_var[features_time_var["event_time_minute"] <= time_point]

    features_time_inv_dataset = features_time_inv_dataset[
        features_time_inv_dataset["stay_id"].isin(features_time_var_dataset["stay_id"].unique())]

    # train:val:test = 3:1:1
    train_val_time_inv, test_time_inv = train_test_split(features_time_inv_dataset, random_state=666, test_size=0.2,
                                                         shuffle=True, stratify=features_time_inv_dataset["output"])
    train_time_inv, val_time_inv = train_test_split(train_val_time_inv, random_state=666, test_size=0.25, shuffle=True,
                                                    stratify=train_val_time_inv["output"])

    train_time_var = features_time_var_dataset[features_time_var_dataset["stay_id"].isin(train_time_inv["stay_id"])]
    val_time_var = features_time_var_dataset[features_time_var_dataset["stay_id"].isin(val_time_inv["stay_id"])]
    test_time_var = features_time_var_dataset[features_time_var_dataset["stay_id"].isin(test_time_inv["stay_id"])]

    # standardize time-invariant features based on training set
    train_time_inv, train_time_var, standardization_statistics = calculate_standardization_statistics(
        train_time_inv.copy(), train_time_var.copy(), dataset_path)
    val_time_inv, val_time_var = apply_standardization_statistics(val_time_inv.copy(), val_time_var.copy(),
                                                                  standardization_statistics)
    test_time_inv, test_time_var = apply_standardization_statistics(test_time_inv.copy(), test_time_var.copy(),
                                                                    standardization_statistics)

    print("Saving patients with time-invariant features!")
    train_time_inv.sample(frac=1, random_state=666).to_csv(os.path.join(dataset_path, "train_feat_time_inv.csv"),
                                                           index=False)
    val_time_inv.sample(frac=1, random_state=666).to_csv(os.path.join(dataset_path, "val_feat_time_inv.csv"),
                                                         index=False)
    test_time_inv.sample(frac=1, random_state=666).to_csv(os.path.join(dataset_path, "test_feat_time_inv.csv"),
                                                          index=False)

    print("Saving patients with time-variant features!")
    train_time_var.to_csv(os.path.join(dataset_path, "train_feat_time_var.csv"), index=False)
    val_time_var.to_csv(os.path.join(dataset_path, "val_feat_time_var.csv"), index=False)
    test_time_var.to_csv(os.path.join(dataset_path, "test_feat_time_var.csv"), index=False)


def calculate_standardization_statistics(train_time_inv, train_time_var, dataset_path):
    # time-invariant
    features_time_inv2standardized = train_time_inv[["adm_age", "weight"]]
    features_time_inv2standardized_mean = features_time_inv2standardized.mean()
    features_time_inv2standardized_std = features_time_inv2standardized.std()
    train_time_inv[["adm_age", "weight"]] = (
                                                    features_time_inv2standardized - features_time_inv2standardized_mean) / features_time_inv2standardized_std

    features_time_inv2standardized_mean.to_csv(os.path.join(dataset_path, "features_time_inv2standardized_mean.csv"),
                                               index=True)
    features_time_inv2standardized_std.to_csv(os.path.join(dataset_path, "features_time_inv2standardized_std.csv"),
                                              index=True)
    # time-var
    features_time_var2standardized = train_time_var[['so2', 'po2', 'pco2', 'pao2fio2ratio', 'ph', 'baseexcess',
                                                     'bicarbonate', 'totalco2', 'hematocrit', 'hemoglobin',
                                                     'carboxyhemoglobin', 'methemoglobin', 'chloride', 'calcium',
                                                     'potassium', 'sodium', 'lactate', 'heart_rate_avg', 'mbp_ni_avg',
                                                     'spo2_avg', 'glucose_new', 'temperature_new', 'dbp_avg_new',
                                                     'fio2_new',
                                                     'aado2_new', 'sbp_avg_new']]

    features_time_var2standardized_mean = features_time_var2standardized.mean()
    features_time_var2standardized_std = features_time_var2standardized.std()

    features_time_var2standardized_mean.to_csv(os.path.join(dataset_path, "features_time_var2standardized_mean.csv"),
                                               index=True)
    features_time_var2standardized_std.to_csv(os.path.join(dataset_path, "features_time_var2standardized_std.csv"),
                                              index=True)

    train_time_var[['so2', 'po2', 'pco2', 'pao2fio2ratio', 'ph', 'baseexcess',
                    'bicarbonate', 'totalco2', 'hematocrit', 'hemoglobin',
                    'carboxyhemoglobin', 'methemoglobin', 'chloride', 'calcium',
                    'potassium', 'sodium', 'lactate', 'heart_rate_avg', 'mbp_ni_avg',
                    'spo2_avg', 'glucose_new', 'temperature_new', 'dbp_avg_new',
                    'fio2_new',
                    'aado2_new', 'sbp_avg_new']] = (
                                                           features_time_var2standardized - features_time_var2standardized_mean) / features_time_var2standardized_std

    return train_time_inv, train_time_var, (
        features_time_inv2standardized_mean, features_time_inv2standardized_std, features_time_var2standardized_mean,
        features_time_var2standardized_std)


def apply_standardization_statistics(time_inv, time_var, standardization_statistics):
    (features_time_inv2standardized_mean, features_time_inv2standardized_std, features_time_var2standardized_mean,
     features_time_var2standardized_std) = standardization_statistics

    # time-invariant
    features_time_inv2standardized = time_inv[["adm_age", "weight"]]
    time_inv[["adm_age", "weight"]] = (
                                              features_time_inv2standardized - features_time_inv2standardized_mean) / features_time_inv2standardized_std

    # time-var
    features_time_var2standardized = time_var[['so2', 'po2', 'pco2', 'pao2fio2ratio', 'ph', 'baseexcess',
                                               'bicarbonate', 'totalco2', 'hematocrit', 'hemoglobin',
                                               'carboxyhemoglobin', 'methemoglobin', 'chloride', 'calcium',
                                               'potassium', 'sodium', 'lactate', 'heart_rate_avg', 'mbp_ni_avg',
                                               'spo2_avg', 'glucose_new', 'temperature_new', 'dbp_avg_new',
                                               'fio2_new',
                                               'aado2_new', 'sbp_avg_new']]

    time_var[['so2', 'po2', 'pco2', 'pao2fio2ratio', 'ph', 'baseexcess',
              'bicarbonate', 'totalco2', 'hematocrit', 'hemoglobin',
              'carboxyhemoglobin', 'methemoglobin', 'chloride', 'calcium',
              'potassium', 'sodium', 'lactate', 'heart_rate_avg', 'mbp_ni_avg',
              'spo2_avg', 'glucose_new', 'temperature_new', 'dbp_avg_new',
              'fio2_new',
              'aado2_new', 'sbp_avg_new']] = (
                                                     features_time_var2standardized - features_time_var2standardized_mean) / features_time_var2standardized_std

    return time_inv, time_var


def generate_exp_name():
    cur_time = datetime.now()
    return "exp_" + cur_time.strftime("%Y%m%d_%H%M%S")


def process_data(data):
    # 1. pre-process the time_invariant features
    features_time_inv = data[["stay_id", 'adm_age', 'weight', 'myocardial_infarct', 'congestive_heart_failure',
                              'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia',
                              'chronic_pulmonary_disease',
                              'rheumatic_disease', 'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
                              'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
                              'severe_liver_disease',
                              'metastatic_solid_tumor', 'aids', 'charlson_comorbidity_index', 'icu_intime',
                              'aki_charttime',
                              "output"]].drop_duplicates()

    # create new column to indicate the aki onset time from the icu intime
    features_time_inv["icu_intime"] = pd.to_datetime(features_time_inv["icu_intime"])
    features_time_inv["aki_charttime"] = pd.to_datetime(features_time_inv["aki_charttime"])
    features_time_inv["aki_onset_minute"] = (
            features_time_inv["aki_charttime"] - features_time_inv["icu_intime"]).astype(
        "timedelta64[m]")
    features_time_inv = features_time_inv.drop(columns=["icu_intime", "aki_charttime"])

    # 2. pre-process time-variant features
    features_time_var = data.copy()[['stay_id', 'icu_intime', 'charttime',
                                     # blood gas
                                     'so2', 'po2', 'pco2', 'pao2fio2ratio', 'ph',
                                     'baseexcess', 'bicarbonate', 'totalco2', 'hematocrit',
                                     'hemoglobin', 'carboxyhemoglobin', 'methemoglobin',
                                     'chloride', 'calcium', 'potassium', 'sodium', 'lactate',
                                     # vital signs
                                     'heart_rate_avg', 'mbp_ni_avg', 'spo2_avg', 'glucose_new',
                                     'temperature_new', 'dbp_avg_new', 'fio2_new', 'aado2_new',
                                     'sbp_avg_new',
                                     # medication
                                     'ace inhibitor', 'acetaminophen', 'alpha adrenergic agonist',
                                     'alpha blocker', 'angiotensin receptor blocker',
                                     'angiotensin receptor neprilysin inhibitor', 'anti-arrhythmic',
                                     'anti-diuretics', 'antibiotics', 'anticoagulant', 'anticonvulsant',
                                     'antidepressant', 'antiemetics', 'antifungals',
                                     'antihypertensive alpha blocker', 'antioxidant', 'antiplatelet',
                                     'antipsychotic', 'antivirals', 'benzodiazepine', 'beta blocker',
                                     'ca supplement', 'calcium channel blocker', 'chemotherapy',
                                     'diuretics', 'fibrates', 'gout medications', 'h2 receptor blocker',
                                     'hmg-coa reductase inhibitors', 'immunosuppressant', 'insulin',
                                     'metformin', 'mood stabilizer', 'nitrates', 'nsaids', 'opioids',
                                     'pde-3 inhibitor', 'phosphate binder', 'proton pump inhibitor',
                                     'sedatives', 'steroids', 'thyroid hormone', 'vasodilator',
                                     'vasopressor']]

    # create a new column for the time of event relative to the ICU admission
    features_time_var["icu_intime"] = pd.to_datetime(features_time_var["icu_intime"])
    features_time_var["charttime"] = pd.to_datetime(features_time_var["charttime"])
    features_time_var["event_time_minute"] = (features_time_var["charttime"] - features_time_var["icu_intime"]).astype(
        "timedelta64[m]")
    features_time_var = features_time_var.drop(columns=["icu_intime", "charttime"])

    return (features_time_inv, features_time_var)


def generate_test_set4transfer_learning(time_point_target, folder2save, features_time_inv, features_time_var):
    # Load test stay_ids
    data_folder = os.path.join(folder2save, f"Dataset_{time_point_target}_{time_point_target + 360}")
    test_stay_ids = pd.read_csv(os.path.join(data_folder, "test_feat_time_inv.csv"))["stay_id"]

    #
    test_feat_time_inv = features_time_inv[features_time_inv["stay_id"].isin(test_stay_ids)]
    features_time_var_dataset = features_time_var[features_time_var["event_time_minute"] <= time_point_target]
    test_feat_time_var = features_time_var_dataset[features_time_var_dataset["stay_id"].isin(test_stay_ids)]

    #
    statistics_folder = os.path.join(folder2save, "Dataset_360_720")
    features_time_inv2standardized_mean = pd.read_csv(
        os.path.join(statistics_folder, "features_time_inv2standardized_mean.csv"), index_col=0, squeeze=True)
    features_time_inv2standardized_std = pd.read_csv(
        os.path.join(statistics_folder, "features_time_inv2standardized_std.csv"), index_col=0, squeeze=True)
    features_time_var2standardized_mean = pd.read_csv(
        os.path.join(statistics_folder, "features_time_var2standardized_mean.csv"), index_col=0, squeeze=True)
    features_time_var2standardized_std = pd.read_csv(
        os.path.join(statistics_folder, "features_time_var2standardized_std.csv"), index_col=0, squeeze=True)

    standardization_statistics = (
        features_time_inv2standardized_mean, features_time_inv2standardized_std, features_time_var2standardized_mean,
        features_time_var2standardized_std)

    #
    test_time_inv, test_time_var = apply_standardization_statistics(test_feat_time_inv.copy(),
                                                                    test_feat_time_var.copy(),
                                                                    standardization_statistics)

    dataset_path = os.path.join(data_folder, "transfer_learning_eval")
    os.makedirs(dataset_path, exist_ok=True)
    test_time_inv.to_csv(os.path.join(dataset_path, "test_feat_time_inv.csv"), index=False)
    test_time_var.to_csv(os.path.join(dataset_path, "test_feat_time_var.csv"), index=False)
