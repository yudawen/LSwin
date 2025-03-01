# Attention
  Download the pretrained weight of Swin Transformer (small version) to the LSwin/models/.
  
  Link is https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth.

  You can get the tip to use the new backbone from LSwin/build_backbone.py

  # Envirenment

  ## Create a conda virtual environment and activate it:
    conda create -n lswin python=3.7 -y
    
    conda activate lswin
  ### Install CUDA>=10.2 with cudnn>=7 following the official installation instructions
  
  ### Install PyTorch>=1.8.0 and torchvision>=0.9.0 with CUDA>=10.2:
  
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
  ### Install timm==0.4.12:
    pip install timm==0.4.12
    
 ### Install other requirements:
    pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
