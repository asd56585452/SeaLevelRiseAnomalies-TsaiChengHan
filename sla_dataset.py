import os
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset


class SLADataset(Dataset):
    def __init__(self, nc_data_dir, label_data_dir, locations, fill_value=0.0):
        """
        初始化 SLADataset。

        参数:
        - nc_data_dir: 存放 .nc 文件的目录路径。
        - label_data_dir: 存放每个站点标签数据的目录路径。
        - locations: 测站地名的列表，用于正确排序标签。
        """
        self.nc_data_dir = nc_data_dir
        self.label_data_dir = label_data_dir
        self.locations = locations  # 保持固定顺序
        self.fill_value = fill_value

        # 提取所有的时间点（从 .nc 文件名中提取日期部分）
        self.all_dates = sorted(
            [
                f.split("_")[2]
                for f in os.listdir(nc_data_dir)
                if f.endswith(".nc")
            ]
        )

        # 加载所有站点的标签数据
        self.labels_data = {}
        for label_file in os.listdir(label_data_dir):
            if label_file.endswith(".csv"):
                # 从文件名提取站点名称（去掉年份和后缀部分）
                location = "_".join(label_file.split("_")[:-4])  # 提取完整站点名称
                location = location.replace("_", " ")  # 替换下划线为空格，保持名称格式一致
                if location in self.locations:
                    file_path = os.path.join(label_data_dir, label_file)
                    label_df = pd.read_csv(file_path)
                    # 将日期列格式化为 YYYYMMDD
                    label_df["t"] = label_df["t"].apply(lambda x: x.replace("-", ""))
                    self.labels_data[location] = label_df

    def __len__(self):
        """返回数据集的大小"""
        return len(self.all_dates)

    def __getitem__(self, idx):
        """
        根据索引返回样本。

        返回:
        - data: 一个 2D numpy 数组，包含 SLA 数据的均值。
        - label: 对应所有站点的异常标记（1 或 0）。
        """
        # 获取当前日期
        date = self.all_dates[idx]

        # 根据日期构建 .nc 文件路径
        file_path = os.path.join(self.nc_data_dir, f"dt_ena_{date}_vDT2021.nc")

        # 读取 .nc 文件中的 SLA 数据
        ds = xr.open_dataset(file_path)
        sla = ds["sla"].values
        ds.close()
        sla = np.nan_to_num(sla, nan=self.fill_value)
        # 构建标签数据
        labels = np.zeros(len(self.locations), dtype=np.int64)  # 默认所有站点为 0（非异常）
        for i, location in enumerate(self.locations):
            if location in self.labels_data:
                location_data = self.labels_data[location]
                # 筛选当前日期的标签（列名 "t"）
                anomaly = location_data[location_data["t"] == date]["anomaly"]
                if not anomaly.empty:
                    labels[i] = anomaly.values[0]
        return sla, labels




from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 定义测站地點
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

    # 数据目录
    nc_data_dir = "F:/SeaLevelRiseAnomalies/Copernicus_ENA_Satelite Maps_Training_Data"
    label_data_dir = "F:/SeaLevelRiseAnomalies/Training_Anomalies_Station Data"

    # 初始化数据集和 DataLoader
    sla_dataset = SLADataset(nc_data_dir, label_data_dir, locations, fill_value=0.0)
    print(sla_dataset.locations)
    sla_loader = DataLoader(sla_dataset, batch_size=8, shuffle=False)  # 单样本加载以便可视化

    # 测试数据加载并可视化
    for batch_idx, (sla, labels) in enumerate(sla_loader):
        print(sla.shape)
        sla = sla[0]  # 提取单个样本的 SLA 数据
        labels = labels[0].numpy()

        print(f"Batch {batch_idx + 1}:")
        print(f"SLA shape: {sla.shape}")  # SLA 数据的形状
        print(f"Labels: {labels}")  # 测站标签

        # 可视化 SLA 数据（假设是 2D 数据）
        plt.figure(figsize=(10, 6))
        plt.contourf(sla[0], cmap="bwr")  # 仅可视化第一个时间步的数据
        plt.colorbar(label="Sea Level Anomaly (m)")
        plt.title(f"SLA Visualization for Date: {sla_dataset.all_dates[batch_idx]}")
        plt.xlabel("Longitude Index")
        plt.ylabel("Latitude Index")
        plt.show()
        break