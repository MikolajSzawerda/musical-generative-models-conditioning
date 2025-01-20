import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tools.project import INPUT_PATH, MODELS_PATH
# -------------------------
# 1) Define the LightningModule
# -------------------------
class LitMNISTModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # output: 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                            # output: 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# output: 64 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),                            # output: 64 x 8 x 8
        )

        self.last_hidden_layer = nn.Linear(64 * 8 * 8, 64)
        
        # Classifier on top of the extracted features
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 10)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        # Flatten for the classifier
        features = features.view(features.size(0), -1)
        embeds = self.last_hidden_layer(features)
        # Get logits
        logits = self.classifier(embeds)
        return logits
    
    def get_features(self, x):
        """
        Returns only the 512-dimensional features (no final classification).
        """
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        feats = self.last_hidden_layer(feats)
        return feats

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# -------------------------
# 2) Define a DataModule for MNIST
# -------------------------
class FilteredCIFAR10(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.indices = []
        for i, (_, label) in enumerate(self.ds):
            if label in [3, 5]:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.ds[real_idx]
        return image, label
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        # Transform to resize 28x28 -> 32x32 and normalize
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        self.mnist_train = FilteredCIFAR10(MNIST(root=INPUT_PATH('cifar'), train=True, download=False, transform=self.transform))
        self.mnist_val = FilteredCIFAR10(MNIST(root=INPUT_PATH('cifar'), train=False, download=False, transform=self.transform))

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

# -------------------------
# 3) Train the model
# -------------------------
if __name__ == "__main__":
    # Create our data module
    dm = MNISTDataModule(batch_size=64)

    # Create our model
    model = LitMNISTModel(lr=1e-3)

    # Create trainer
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",  # "gpu" if you have a GPU
        devices="auto"       # or specify device index
    )

    # Fit / train the model
    trainer.fit(model, dm)

    # -------------------------
    # 4) Save the trained weights
    # -------------------------
    # Saving only the model's state_dict (weights)
    torch.save(model.state_dict(), MODELS_PATH('minist', "mnist_feature_extractor_weights.pth"))

    # Alternatively, save the entire checkpoint (includes optimizer state, etc.)
    # trainer.save_checkpoint("mnist_feature_extractor.ckpt")
