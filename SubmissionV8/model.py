# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd

# class Model:
#     def __init__(self):
#         print("Dummy Model initialized")
#         # Put your initialization code here
#         # load the save model here
        

#     def predict(self, X):
#         # 要處裡nan
#         # Put your prediction code here
#         # This example predicts a random value for 12 station
#         # The output should be a dataframe with 10 rows and 12 columns
#         # Each value should be 1 for anamoly and 0 for normal
#         # Return a np array of 1s and 0s with the same length of 12
#         # with random prediction of 1 or 0
#         return torch.randint(0, 2, (12,)).numpy()
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import mode  # 统计众数
import os

class Model:
    def __init__(self):
        print("Loading trained model...")
        # Load the saved model
        self.num_classes = 12
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        dropout_rate = 0.11218913608873872
        conv1_out_channels = 48
        conv2_out_channels = 96
        conv3_out_channels = 128
        self.fold_count = 5
        self.models = []
        for fold in range(1, self.fold_count + 1):
          model = DynamicCNN(conv1_out_channels, conv2_out_channels, conv3_out_channels, dropout_rate, num_classes=len(locations)).to(self.device)
          model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), str(fold)+'best_sla_cnn_model.pth'), map_location=self.device))
          model.eval()
          self.models.append(model)

    def preprocess(self, X):
        # Replace NaN values with 0
        X = np.nan_to_num(X, nan=0.0)
        # Ensure the shape is compatible with the model
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch and channel dims
        return X

    def predict(self, X):
        # Preprocess the input
        X_processed = self.preprocess(X)
        # Make predictions
        preds = []
        for fold in range(1, self.fold_count + 1):
          with torch.no_grad():
              outputs = self.models[fold-1](X_processed)
              pred = torch.sigmoid(outputs).cpu().numpy() > 0.5  # Binary prediction
              pred = pred.reshape(1,12)
              preds.append(pred)
        preds = np.concatenate(preds, axis=0).astype(np.int64)
        return mode(preds, axis=0).mode.flatten().reshape((1, 12))

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
        