# Yolo model optimizations on OrangePi

Created: August 3, 2024 11:40 AM
Class: Embedded AI

# Homework 3 Report

## .pt 2 onnx

This conversion usually works out of the box as it did this time:

```jsx
yolo export model=yolov8n.pt format=onnx imgsz=640

```

![Знімок екрана 2024-08-03 о 11.56.45.png](Yolo%20model%20optimizations%20on%20OrangePi%208965a6a9925d4d1482220c67a2aa990f/%25D0%2597%25D0%25BD%25D1%2596%25D0%25BC%25D0%25BE%25D0%25BA_%25D0%25B5%25D0%25BA%25D1%2580%25D0%25B0%25D0%25BD%25D0%25B0_2024-08-03_%25D0%25BE_11.56.45.png)

## Onnx 2 TFLite

```jsx
onnx2tf -i "yolov8n.onnx" -o "yolov8n_tflite" -nuo
```

![Знімок екрана 2024-08-03 о 11.41.58.png](Yolo%20model%20optimizations%20on%20OrangePi%208965a6a9925d4d1482220c67a2aa990f/%25D0%2597%25D0%25BD%25D1%2596%25D0%25BC%25D0%25BE%25D0%25BA_%25D0%25B5%25D0%25BA%25D1%2580%25D0%25B0%25D0%25BD%25D0%25B0_2024-08-03_%25D0%25BE_11.41.58.png)

## .pt 2 TFLite (More complex method dependency-wise)

```jsx
yolo export model=yolov8n.pt format=tflite imgsz=640
```

![Знімок екрана 2024-08-03 о 11.52.25.png](Yolo%20model%20optimizations%20on%20OrangePi%208965a6a9925d4d1482220c67a2aa990f/%25D0%2597%25D0%25BD%25D1%2596%25D0%25BC%25D0%25BE%25D0%25BA_%25D0%25B5%25D0%25BA%25D1%2580%25D0%25B0%25D0%25BD%25D0%25B0_2024-08-03_%25D0%25BE_11.52.25.png)

In order to make this work I had to downgrade tflite (The precompiled Tensorflow package wants a newer libstdc++ than is provided with Bullseye)

```jsx
python3 -m pip install --upgrade tflite-support==0.4.2
python3 -m pip install --upgrade tflite-runtime==2.11.0
```

# Onnx to RKNN on RK3588

```jsx
git clone https://github.com/airockchip/rknn_model_zoo
cd rknn_model_zoo/examples/yolov8/model
bash download_model.sh

cd ../python/
git clone https://github.com/airockchip/rknn-toolkit2/
```

![Знімок екрана 2024-08-03 о 12.32.49.png](Yolo%20model%20optimizations%20on%20OrangePi%208965a6a9925d4d1482220c67a2aa990f/%25D0%2597%25D0%25BD%25D1%2596%25D0%25BC%25D0%25BE%25D0%25BA_%25D0%25B5%25D0%25BA%25D1%2580%25D0%25B0%25D0%25BD%25D0%25B0_2024-08-03_%25D0%25BE_12.32.49.png)

# Inference speed comparison

| Model | Average Inference Time, ms | Average FPS |
| --- | --- | --- |
| yolov8n_float32.tflite | 520 | 1,92 |
| yolov8n_float16.tflite | 500 | 2 |
| yolov8n.pt | 410 | 2,4 |
| yolov8n.onnx | 275 | 3,6 |
| yolov8n.rknn | 42 | 24 |