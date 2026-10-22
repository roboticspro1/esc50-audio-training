#!/usr/bin/env python
# coding: utf-8

# # 测试 AudioNet
# 
# 先运行 `train.ipynb`，生成 `outputs/audionet.ckpt`。
# 本笔记本只加载该检查点，在 fold 5 上计算准确率，并把每条预测写到 `outputs/test_predictions.csv`。
# 

# In[ ]:


import torch, torchaudio
from torch import nn
from torch.nn import functional as F


# In[ ]:


import pytorch_lightning as pl
from torchmetrics.functional import accuracy


# In[ ]:


from pathlib import Path
import pandas as pd


# In[ ]:


torch.cuda.is_available()


# In[ ]:


class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50

    def __init__(self, path: Path = Path('ESC-50-master/ESC-50-master'),
                 sample_rate: int = 8000,
                 folds = [1]):
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.path = path
        self.csv = pd.read_csv(path / Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / 'audio' / row['filename'])
        label = row['target']
        xb = self.db(
            self.melspec(
                self.resample(wav)
            )
        )
        return xb, label

    def __len__(self):
        # Returns length
        return len(self.csv)


# In[ ]:


class AudioNet(pl.LightningModule):

    def __init__(self, n_classes = 50, base_filters = 32):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(1, base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(base_filters, base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 2)
        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(base_filters * 4, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=50)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        return acc


# In[ ]:


ckpt_path = Path("outputs/audionet-best.ckpt")
audionet = AudioNet.load_from_checkpoint(ckpt_path)


# In[ ]:


test_data = ESC50Dataset(folds=[5])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)
len(test_data)


# In[ ]:


trainer = pl.Trainer(accelerator="auto", devices=1)
trainer.test(audionet, test_loader)


# In[ ]:


audionet.eval()
predicted = []
with torch.no_grad():
    for xb, _ in test_loader:
        logits = audionet(xb.to(audionet.device))
        predicted.append(logits.argmax(dim=1).cpu())
predicted = torch.cat(predicted).numpy()

meta = test_data.csv.reset_index(drop=True)
label_names = (
    pd.read_csv(test_data.path / "meta/esc50.csv")[["target", "category"]]
    .drop_duplicates()
    .set_index("target")["category"]
)
predictions = pd.DataFrame({
    "filename": meta["filename"].to_numpy(),
    "target": meta["target"].to_numpy(),
    "prediction": predicted,
})
predictions["category"] = predictions["target"].map(label_names)
predictions["predicted_category"] = predictions["prediction"].map(label_names)
predictions["correct"] = predictions["target"].to_numpy() == predictions["prediction"].to_numpy()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
pred_path = output_dir / "test_predictions_best.csv"
predictions.to_csv(pred_path, index=False)

print(f"predictions: {pred_path.resolve()}")
print(f"test accuracy: {predictions['correct'].mean():.3f} ({predictions['correct'].sum()}/{len(predictions)})")
predictions.head(10)


# In[ ]:




