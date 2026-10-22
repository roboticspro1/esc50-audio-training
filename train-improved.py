#!/usr/bin/env python
# coding: utf-8

# # 训练 AudioNet
# 
# 用 ESC-50 的 fold 1–3 训练，fold 4 做验证。训练结束后模型写到 `outputs/audionet.ckpt`。
# 测试请运行 `test.ipynb`，不要在这里使用 fold 5。
# 

# ## Reproducible Deep Learning (PhD course, Data Science)
# ### Lecture 1: deep learning recap
# 
# A convolutional neural network for the ESC-50 dataset: https://github.com/karolpiczak/ESC-50.
# 

# In[ ]:


import torch, torchaudio
from torch import nn
from torch.nn import functional as F


# In[ ]:


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.functional import accuracy


# In[ ]:


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


torch.cuda.is_available()


# ### Step 1: Some experiments in audio loading and transformation
# 
# The code in this section is just for experimentation.
# 

# In[ ]:


# Root folder of ESC-50, containing the subfolders 'audio' and 'meta'.
datapath = Path('ESC-50-master/ESC-50-master')


# In[ ]:


datapath.exists()


# In[ ]:


csv = pd.read_csv(datapath / Path('meta/esc50.csv'))
csv.head()


# In[ ]:


x, sr = torchaudio.load(datapath / 'audio' / csv.iloc[0, 0], normalize=True)
x.shape


# In[ ]:


plt.plot(x[0, ::5])


# In[ ]:


h = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(x)
h = torchaudio.transforms.AmplitudeToDB()(h)
plt.imshow(h[0])


# ### Step 2: Putting together data loading and preprocessing
# 

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


train_data = ESC50Dataset(folds=[1, 2, 3])
val_data = ESC50Dataset(folds=[4])
len(train_data), len(val_data)


# In[ ]:


xb, yb = train_data[0]
xb.shape, yb


# ### Step 3: Build a classification model and train
# 

# In[ ]:


train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)


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

    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=50)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        return optimizer


# In[ ]:


pl.seed_everything(0)


# In[ ]:


# Check that the network works on a single mini-batch.
audionet = AudioNet()
xb, yb = next(iter(train_loader))
audionet(xb).shape


# In[ ]:


output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath="outputs",
    filename="audionet-best",
    monitor="val_acc",
    mode="max",
    save_top_k=1,
)

early_stop = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=8,
)

trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stop],
)

trainer.fit(audionet, train_loader, val_loader)

print("Best model:", checkpoint_callback.best_model_path)
print("Best validation accuracy:", checkpoint_callback.best_model_score)


# In[ ]:


output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir,
    filename="best-audionet",
    monitor="val_acc",
    mode="max",
    save_top_k=1,
)

early_stop_callback = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=10,
)

trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stop_callback],
)

trainer.fit(audionet, train_loader, val_loader)


# In[ ]:


output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

ckpt_path = output_dir / "audionet-improved-last.ckpt"
trainer.save_checkpoint(ckpt_path)
print(ckpt_path.resolve())

