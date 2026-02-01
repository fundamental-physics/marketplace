# Embedding Networks for High-Dimensional Observations

## Table of Contents

1. [Overview](#overview)
2. [When to Use Embeddings](#when-to-use-embeddings)
3. [Built-in Embedding Support](#built-in-embedding-support)
4. [CNN Embeddings for Images](#cnn-embeddings-for-images)
5. [RNN Embeddings for Time Series](#rnn-embeddings-for-time-series)
6. [Custom Embeddings](#custom-embeddings)
7. [Training Considerations](#training-considerations)

---

## Overview

Embedding networks transform high-dimensional observations into lower-dimensional summary statistics before density estimation. The embedding is learned jointly with the density estimator during training.

```
x (high-dim) → Embedding Network → z (low-dim) → Density Estimator → p(θ|z)
```

This is essential when observations are:
- Images (2D spatial data)
- Time series (sequential data)
- Spectra (1D structured data)
- Any data with dimension > ~50-100

---

## When to Use Embeddings

| Observation Type | Dimension | Recommendation |
|-----------------|-----------|----------------|
| Low-dimensional summary stats | < 20 | No embedding needed |
| Moderate vectors | 20-100 | Optional, try without first |
| High-dimensional vectors | > 100 | MLP embedding |
| Images | H × W | CNN embedding |
| Time series | T × features | RNN/1D-CNN embedding |
| Graphs | Variable | GNN embedding |

**Rule of thumb:** If direct density estimation fails or produces poor posteriors, try adding an embedding network.

---

## Built-in Embedding Support

All inference classes support the `embedding_net` parameter through density estimator configuration:

```python
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# Create density estimator with embedding
density_estimator = posterior_nn(
    model="nsf",
    embedding_net=my_embedding_network,  # Your PyTorch nn.Module
)

inference = NPE(prior=prior, density_estimator=density_estimator)
```

The embedding network must:
- Be a `torch.nn.Module`
- Accept observations of shape `(batch, *obs_shape)`
- Output embeddings of shape `(batch, embedding_dim)`

---

## CNN Embeddings for Images

### Basic CNN for 2D Images

```python
import torch
import torch.nn as nn
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

class CNNEmbedding(nn.Module):
    def __init__(self, input_channels=1, image_size=64, embedding_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        # Calculate flattened size
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        features = self.conv(x)
        return self.fc(features)

# Usage
embedding = CNNEmbedding(input_channels=1, image_size=64, embedding_dim=64)
density_estimator = posterior_nn(model="nsf", embedding_net=embedding)
inference = NPE(prior=prior, density_estimator=density_estimator)
```

### ResNet-style Embedding

For more complex images:

```python
class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Use pretrained ResNet as feature extractor
        resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # x: (batch, 3, H, W) - ResNet expects 3 channels
        features = self.features(x).flatten(1)
        return self.fc(features)
```

---

## RNN Embeddings for Time Series

### GRU for Sequential Data

```python
class GRUEmbedding(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, embedding_dim=32, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Bidirectional doubles the hidden dim
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        # x shape: (batch, time_steps, features)
        _, hidden = self.gru(x)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)

# Usage for gravitational wave strain data
embedding = GRUEmbedding(input_dim=1, hidden_dim=64, embedding_dim=32)
density_estimator = posterior_nn(model="nsf", embedding_net=embedding)
```

### 1D CNN for Time Series

Often faster than RNNs:

```python
class Conv1DEmbedding(nn.Module):
    def __init__(self, input_channels=1, seq_length=1024, embedding_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 8, embedding_dim)

    def forward(self, x):
        # x shape: (batch, channels, time)
        features = self.conv(x)
        return self.fc(features)
```

---

## Custom Embeddings

### Combining Multiple Data Types

```python
class MultiModalEmbedding(nn.Module):
    def __init__(self, image_embedding_dim=32, spectrum_embedding_dim=32):
        super().__init__()
        # Image branch
        self.image_embed = CNNEmbedding(embedding_dim=image_embedding_dim)
        # Spectrum branch
        self.spectrum_embed = Conv1DEmbedding(embedding_dim=spectrum_embedding_dim)

    def forward(self, x):
        # x is a dict or tuple with multiple observation types
        image, spectrum = x['image'], x['spectrum']
        img_emb = self.image_embed(image)
        spec_emb = self.spectrum_embed(spectrum)
        return torch.cat([img_emb, spec_emb], dim=1)
```

### Physics-Informed Embeddings

Incorporate domain knowledge:

```python
class PhysicsEmbedding(nn.Module):
    def __init__(self, raw_dim, embedding_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(raw_dim + 3, 64),  # +3 for physics features
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        # Compute physics-motivated summary statistics
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        max_val = x.max(dim=-1, keepdim=True)[0]

        # Concatenate with raw data
        x_augmented = torch.cat([x, mean, std, max_val], dim=-1)
        return self.fc(x_augmented)
```

---

## Training Considerations

### Embedding Dimension Selection

| Observation Complexity | Recommended Embedding Dim |
|-----------------------|--------------------------|
| Simple structure | 16-32 |
| Moderate complexity | 32-64 |
| Complex (images, long time series) | 64-128 |
| Very high information content | 128-256 |

**Rule:** Embedding dim should be >= number of parameters you're inferring, but not excessively large.

### Preventing Overfitting

```python
class RegularizedEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)
```

### Pre-training Embeddings

For complex observations, pre-train the embedding:

```python
# Option 1: Autoencoder pre-training
class Autoencoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.encoder = embedding
        self.decoder = nn.Linear(embedding.fc.out_features, embedding.fc.in_features)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Train autoencoder on observations only (no labels needed)
autoencoder = Autoencoder(embedding)
# ... train to minimize reconstruction loss ...

# Then use pre-trained encoder
density_estimator = posterior_nn(model="nsf", embedding_net=autoencoder.encoder)
```

### Freezing vs. Joint Training

```python
# Option 1: Train embedding jointly (default, usually best)
density_estimator = posterior_nn(model="nsf", embedding_net=embedding)

# Option 2: Freeze pre-trained embedding
for param in embedding.parameters():
    param.requires_grad = False
density_estimator = posterior_nn(model="nsf", embedding_net=embedding)
```

Joint training is usually preferred unless the embedding was pre-trained on a large dataset.
