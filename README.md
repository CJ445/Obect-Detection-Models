# Obect-Detection-Models

```
git pull https://github.com/CJ445/Object-Detection-Models.git
cd Object-Detection-Models
```

# Miniconda Setup on Ubuntu
## Step 1: Download Miniconda
Open your terminal and download the latest Miniconda installer for Linux (64-bit):
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
## Step 2: Install Miniconda
Run the downloaded installer script:
```
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the prompts:
- Press Enter to review the license agreement.
- Type yes to accept the terms.
- Choose the installation directory (or press Enter to install in the default location ~/miniconda3).
- Type yes to initialize Miniconda.

Once the installation completes, restart the terminal or run the following command to activate Miniconda:
```
source ~/.bashrc
```

# Create a new conda environment named obj-detect
```
conda create -n obj-detect python=3.9 -y
```
# Activate the new environment
```
conda activate obj-detect
```

```
pip install -r requirements.txt
```

# Cascade R-CNN
```
python cascade_rcnn_detection.py
```
# EfficientDet
```
python efficientdet_detection.py
```
# Faster R-CNN
```
python faster_rcnn_detection.py
```
# Mask R-CNN
```
python mask_rcnn_detection.py
```
# RetinaNet
```
python retinanet_detection.py
```
# Sparse R-CNN
```
python sparse_rcnn_detection.py
```
# YOLO-NAS
```
python yolonas_detection.py
```
