# %%
import wandb
from tqdm import tqdm

from models import *
from utils import *

# %% Param prep
logger_mode = 1
# Note: you need to apply for an account at https://wandb.ai first
wandb.init(project="AKI_prediction",
           name=generate_exp_name(),
           save_code=True,
           mode="online" if logger_mode else "disabled",
           # Hyper-parameters and run metadata
           config={
               "learning_rate": 5e-5,
               "max_epochs": 20,
               "scheduler_step_size": 5,
               "scheduler_gamma": 0.7,
               "gpu_on": True,
               "model_name": "gru_fusion",
               "data_dir": "./Data/modeling/Dataset_360_720",
               "time_var_input_dim": 70,
               "time_var_hidden_dim": 20,
               "time_inv_input_dim": 20,
               "time_inv_hidden_dim": 10
           })
config = wandb.config
print(config)
check_train_every_n_epoch = 1
check_val_every_n_epoch = 1
check_test_every_n_epoch = 1
save_model_every_n_epoch = 1
n_classes = 2
train_batch_size = 1
val_batch_size = 1
test_batch_size = 1
num_workers = 32
# Model path
MODELS_FOLDER = os.path.join(config.data_dir, "models")
model_path = os.path.join(MODELS_FOLDER, config.model_name)
os.makedirs(model_path, exist_ok=True)
# Device configuration
device = torch.device("cuda:0" if config.gpu_on else "cpu")

# %% Dataset
data_time_var_path = os.path.join(config.data_dir, "train_feat_time_var.csv")
data_time_inv_path = os.path.join(config.data_dir, "train_feat_time_inv.csv")
train_dataset = MIMIC4Classification(data_time_inv_path, data_time_var_path)

data_time_var_path = os.path.join(config.data_dir, "val_feat_time_var.csv")
data_time_inv_path = os.path.join(config.data_dir, "val_feat_time_inv.csv")
val_dataset = MIMIC4Classification(data_time_inv_path, data_time_var_path)

data_time_var_path = os.path.join(config.data_dir, "test_feat_time_var.csv")
data_time_inv_path = os.path.join(config.data_dir, "test_feat_time_inv.csv")
test_dataset = MIMIC4Classification(data_time_inv_path, data_time_var_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                           num_workers=num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size,
                                         num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          num_workers=num_workers, pin_memory=True)
print(f"Training set: {len(train_dataset)}")
print(f"Val set: {len(val_dataset)}")
print(f"Test set: {len(test_dataset)}")

# %% Model init
fusion_model = GRUFusion(config.time_var_input_dim, config.time_var_hidden_dim, config.time_inv_input_dim,
                         config.time_inv_hidden_dim)
fusion_model.to(device)
# Optimizer
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=config.learning_rate)
# Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size,
                                            gamma=config.scheduler_gamma)
# Loss
criterion = torch.nn.CrossEntropyLoss()

# %% Training
wandb.watch(fusion_model, log="all", log_graph=True)
print("Training starts...")
try:
    for epoch in tqdm(range(config.max_epochs)):
        # Train
        fusion_model.train()
        for (time_inv, time_var, label) in train_loader:
            time_inv = time_inv.to(device)
            time_var = time_var.to(device)
            label = label.long().to(device)

            # Forward pass
            y_pred = fusion_model(time_inv, time_var)

            # Compute loss
            loss = criterion(y_pred, label)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Eval on train
        if epoch % check_train_every_n_epoch == 0:
            train_loss, (auc_roc, f1, acc) = eval_classification(fusion_model, train_loader, device, criterion,
                                                                 report_metrics=True)
            wandb.log({"Loss/train": train_loss, "AUC-ROC/train": auc_roc, "F1 score/train": f1, "Accuracy/train": acc},
                      step=epoch)

        # Eval on val
        if epoch % check_val_every_n_epoch == 0:
            val_loss, (auc_roc, f1, acc) = eval_classification(fusion_model, val_loader, device, criterion,
                                                               report_metrics=True)
            wandb.log({"Loss/val": val_loss, "AUC-ROC/val": auc_roc, "F1 score/val": f1, "Accuracy/val": acc},
                      step=epoch)

        # Eval on 1
        if epoch % check_test_every_n_epoch == 0:
            test_loss, (auc_roc, f1, acc) = eval_classification(fusion_model, test_loader, device, criterion,
                                                                report_metrics=True)
            wandb.log({"Loss/test": test_loss, "AUC-ROC/test": auc_roc, "F1 score/test": f1, "Accuracy/test": acc},
                      step=epoch)

        # checkpoints
        if epoch % save_model_every_n_epoch == 0:
            torch.save({
                'epoch': epoch,
                "fusion_model_state_dict": fusion_model.state_dict(),
            }, os.path.join(model_path, str(epoch) + ".ckpt"))
    wandb.finish()
except Exception as e:
    print(e)

# %%
wandb.finish()
