import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
from sklearn.metrics import f1_score
import random
import numpy as np
from sla_dataset_GPU import SLADataset  # 确保这个文件名与您的数据集文件名一致

# 定義模型架構
class DynamicCNN(nn.Module):
    def __init__(self, conv1_out, conv2_out, conv3_out, dropout_rate, num_classes):
        super(DynamicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=3, padding=1)
        self.fc = nn.Linear(conv3_out * 12 * 20, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 定義評估函數
def evaluate_model(model, loader, criterion, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sla, labels in loader:
            if sla.device != torch.device(device) or sla.dtype != torch.float32:
                sla = sla.to(device, dtype=torch.float32)
            if labels.device != torch.device(device) or labels.dtype != torch.float32:
                labels = labels.to(device, dtype=torch.float32)
            outputs = model(sla)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return f1_score(all_labels, all_preds, average="macro")

# 定義訓練與搜索函數
def train_and_evaluate(trial):
    # Optuna 調整的超參數
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    conv1_out_channels = trial.suggest_int('conv1_out_channels', 8, 64, step=8)
    conv2_out_channels = trial.suggest_int('conv2_out_channels', 16, 128, step=16)
    conv3_out_channels = trial.suggest_int('conv3_out_channels', 32, 256, step=32)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    set_seed(42)

    # 資料加載與模型初始化
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = DynamicCNN(conv1_out_channels, conv2_out_channels, conv3_out_channels, dropout_rate, num_classes=len(locations)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    # 早停參數
    patience = 5
    best_f1 = 0.0
    counter = 0

    for epoch in range(50):  # 設定最大 Epoch
        model.train()
        for sla, labels in train_loader:
            if sla.device != torch.device(device) or sla.dtype != torch.float32:
                sla = sla.to(device, dtype=torch.float32)
            if labels.device != torch.device(device) or labels.dtype != torch.float32:
                labels = labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(sla)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 測試模型
        current_f1 = evaluate_model(model, test_loader, criterion, device)
        trial.report(current_f1, epoch)  # Optuna 報告當前 F1-Score

        # 儲存最佳結果
        if current_f1 > best_f1:
            best_f1 = current_f1
            counter = 0
        else:
            counter += 1

        # 早停判斷
        if counter >= patience:
            break

    return best_f1

# 設置亂數種子
def set_seed(seed=42):
    random.seed(seed)  # Python 內建隨機數生成器
    np.random.seed(seed)  # NumPy 隨機數生成器
    torch.manual_seed(seed)  # PyTorch CPU 隨機數生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU 隨機數生成器
        torch.cuda.manual_seed_all(seed)  # 多 GPU 的隨機數生成器

# 設置 Optuna 搜索
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positive_weight = torch.tensor([
            23.8553, 42.5896, 15.6886, 65.3750, 24.5066, 232.6400,
            1459.2500, 90.2656, 22.1786, 65.3750, 152.7105, 16.5405
        ], device='cuda:0', dtype=torch.float64)
    locations = [
            "Atlantic City",
            "Baltimore",
            "Eastport",
            "Fort Pulaski",
            "Lewes",
            "New London",
            "Newport",
            "Portland",
            "Sandy Hook",
            "Sewells Point",
            "The Battery",
            "Washington",
        ]
    
    set_seed(42)
    
    # 数据目录
    nc_data_dir = "F:/SeaLevelRiseAnomalies/Copernicus_ENA_Satelite Maps_Training_Data"
    label_data_dir = "F:/SeaLevelRiseAnomalies/Training_Anomalies_Station Data"
    train_split = 0.8  # 训练集占比

    # 初始化数据集
    sla_dataset = SLADataset(nc_data_dir, label_data_dir, locations, fill_value=0.0)

    # 切分训练集和测试集
    dataset_size = len(sla_dataset)
    train_size = int(train_split * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(sla_dataset, [train_size, test_size])

    # 使用 SQLite 儲存進度
    storage = "sqlite:///optuna_study.db"
    study_name = "GPU_Dataset_study"  # 定義 Study 名稱 "example_study",
    study = optuna.create_study(storage=storage, study_name=study_name, load_if_exists=True, direction="maximize")

    # 啟動優化，設置運行時間或試驗數量
    study.optimize(train_and_evaluate, timeout=3600*8)  # 運行 1 小時

    # 輸出最佳結果
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best F1-Score: {study.best_value}")

    # 繪製結果
    from optuna.visualization import plot_optimization_history, plot_param_importances
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
