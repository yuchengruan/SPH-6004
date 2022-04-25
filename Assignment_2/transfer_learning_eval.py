import wandb

from models import *
from utils import *

# %% Param prep
logger_mode = 0
# Note: you need to apply for an account at https://wandb.ai first
wandb.init(project="AKI_prediction",
           name=generate_exp_name(),
           save_code=True,
           mode="online" if logger_mode else "disabled",
           # Hyper-parameters and run metadata
           config={
               "gpu_on": True,
               "model_name": "gru_fusion",
               "model_dir": "./Data/modeling/Dataset_360_720",
               "data_dir": "./Data/modeling/Dataset_1080_1440/transfer_learning_eval",
               "time_var_input_dim": 70,
               "time_var_hidden_dim": 20,
               "time_inv_input_dim": 20,
               "time_inv_hidden_dim": 10
           })

config = wandb.config
print(config)
n_classes = 2
train_batch_size = 1
val_batch_size = 1
test_batch_size = 1
num_workers = 32
# Model path
MODELS_FOLDER = os.path.join(config.model_dir, "models")
model_path = os.path.join(MODELS_FOLDER, config.model_name, "19.ckpt")
# Device configuration
device = torch.device("cuda:0" if config.gpu_on else "cpu")

# %% Dataset
data_time_var_path = os.path.join(config.data_dir, "test_feat_time_var.csv")
data_time_inv_path = os.path.join(config.data_dir, "test_feat_time_inv.csv")
test_dataset = MIMIC4Classification(data_time_inv_path, data_time_var_path)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          num_workers=num_workers, pin_memory=True)
print(f"Test set: {len(test_dataset)}")
# Model init
fusion_model = GRUFusion(config.time_var_input_dim, config.time_var_hidden_dim, config.time_inv_input_dim,
                         config.time_inv_hidden_dim)
fusion_model.load_state_dict(torch.load(model_path)["fusion_model_state_dict"])
fusion_model.to(device)

_, (auc_roc, f1, acc) = eval_classification(fusion_model, test_loader, device, report_loss=False,
                                            report_metrics=True)
with open(generate_exp_name() + ".log", "w") as f:
    print(f"{config.data_dir}", file=f)
    print(f"AUC-ROC: {auc_roc}", file=f)
