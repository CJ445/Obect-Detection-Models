# Obect-Detection-Models

```
git pull https://github.com/CJ445/Object-Detection-Models.git
cd Object-Detection-Models
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
python cascade_rcnn_detection.py --model_path cascade_rcnn_detection.py --image_path test.jpeg
```
# EfficientDet
```
python efficientdet_detection.py --model_path efficientdet_detection.py --image_path test.jpeg
```
# Faster R-CNN
```
python faster_rcnn_detection.py --model_path faster_rcnn_detection.py --image_path test.jpeg
```
# Mask R-CNN
```
python mask_rcnn_detection.py --model_path mask_rcnn_detection.py --image_path test.jpeg
```
# RetinaNet
```
python retinanet_detection.py --model_path retinanet_detection.py --image_path test.jpeg
```
# Sparse R-CNN
```
python sparse_rcnn_detection.py --model_path sparse_rcnn_detection.py --image_path test.jpeg
```
# YOLO-NAS
```
python yolonas_detection.py --model_path yolonas_detection.py --image_path test.jpeg
```
