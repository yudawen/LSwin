
# LSwin

LSwin is a lightweight backbone based on the Swin Transformer and designed for remote sensing image interpretation.

## Pretrained Weights

Download the pretrained weights of the **Swin Transformer-Small** model and place the file in:

```text
LSwin/models/
````

The pretrained weights can be downloaded from the following link:

[Download `swin_small_patch4_window7_224_22k.pth`](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth)

For instructions on how to use LSwin as a backbone in your own models, please refer to:

```text
LSwin/build_backbone.py
```

## Environment

### 1. Create a Conda Environment

Create and activate a new Conda environment:

```bash
conda create -n lswin python=3.7 -y
conda activate lswin
```

### 2. Install CUDA

Install **CUDA >= 11.1** according to the official CUDA installation instructions.

### 3. Install PyTorch and torchvision

Install **PyTorch >= 1.11.0** and **torchvision >= 0.9.1** with CUDA >= 11.1.

For example:

```bash
conda install pytorch==1.11.0 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch
```

### 4. Install timm

Install `timm==0.6.11`:

```bash
pip install timm==0.6.11
```

### 5. Install Other Dependencies

Install the remaining dependencies:

```bash
pip install opencv-python==4.1.2.30 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

## Example: Using LSwin as a Backbone

The following example demonstrates how to initialize the LSwin backbone and extract features from an input tensor.

```python
import torch
from models.LSwin_backbone import LSwin_backbone

# Initialize the LSwin backbone
backbone = LSwin_backbone()

# Create a dummy input tensor
# Shape: B × C × H × W
inputs = torch.randn(2, 3, 512, 512)

# Move the model and input data to the device
device = torch.device("cpu")
backbone = backbone.to(device)
inputs = inputs.to(device)

# Extract features
features = backbone(inputs)

# Print the output feature size
print("Output feature size of the backbone:")
print(features.size())  # B × C × H × W
```

You can replace the dummy input with your own remote sensing images and integrate the LSwin backbone into your downstream tasks.


