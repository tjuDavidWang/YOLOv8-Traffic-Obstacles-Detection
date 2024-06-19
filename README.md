# Traffic Obstacle Detection with YOLOv8

## Environment Setup

Ensure your Python version is 3.9.18 and install the following dependencies:

```bash
pip install ultralytics
pip install PyQt6 qt_material opencv-python
```

It's recommended to install `ultralytics` first, then uninstall it to use the project's custom code:

```bash
pip uninstall ultralytics
```

Project directory structure:

```bash
.
├─model       # Pretrained models
├─UI          # Software UI
└─ultralytics # Model code
```

## Training

Our training dataset is from the [Obstacle Dataset](https://github.com/TW0521/Obstacle-Dataset), filtered to 7915 images suitable for autonomous driving scenarios. The dataset includes 15 obstacle types such as "stop_sign", "person", "bicycle", "bus", "truck", "car", "motorbike", "reflective_cone", "ashcan", "warning_column", "spherical_roadblock", "pole", "dog", "tricycle", "fire_hydrant", etc.

To improve the robustness of our model, we used [Albumentations](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/) for jitter augmentation on the bounding boxes in our dataset.

To train the final model, run:

```bash
python train.py
```

To optimize our model for different aspects:
- **SlimNeck**: Used to make the model lightweight, suitable for vehicle-mounted scenarios.
- **GAM**: Used to capture more features.
- **WIOU**: Used to handle inaccuracies in the dataset annotations.

Performance comparison of models:

| Model              | mAP50-95↑ | mAP50↑   | **Recall**↑ | **Precision**↑ | GFLOPs↓  | Param↓      |
| ------------------ | --------- | -------- | ----------- | -------------- | -------- | ----------- |
| Baseline(YOLOV8s)  | 58.5      | 71.9     | 71.8        | 81.2           | 28.4     | 11126358    |
| +SlimNeck          | 59.3      | 72.6     | 71.1        | 82.1           | **23.7** | **9858550** |
| +GAM               | 59.8      | 79.9     | 72.7        | 84.5           | 29.8     | 12865622    |
| +WIOU              | 60.2      | 80.4     | 72.5        | 85.8           | 28.4     | 11126358    |
| +SlimNeck+GAM      | 60.0      | 80.2     | 73.2        | 84.1           | 25.6     | 11728886    |
| +SlimNeck+WIOU     | 59.7      | 79.6     | 72.5        | 84.0           | **23.7** | **9858550** |
| +GAM+WIOU          | 61.2      | 81.8     | 74.4        | 86.6           | 29.8     | 12865622    |
| +SlimNeck+GAM+WIOU | **62.2**  | **83.2** | **75.6**    | **89.0**       | 25.6     | 11728886    |

## Software

To run the software UI:

```bash
cd UI
python main.py
```

## References

1. Sohan M, Sai Ram T, Reddy R, et al. A Review on YOLOv8 and Its Advancements[C]//International Conference on Data Intelligence and Cognitive Informatics. Springer, Singapore, 2024: 529-545.
2. Li H, Li J, Wei H, et al. Slim-neck by GSConv: A better design paradigm of detector architectures for autonomous vehicles[J]. arXiv preprint arXiv:2206.02424, 2022.
3. Liu Y, Shao Z, Hoffmann N. Global attention mechanism: Retain information to enhance channel-spatial interactions[J]. arXiv preprint arXiv:2112.05561, 2021.
4. Tong Z, Chen Y, Xu Z, et al. Wise-IoU: bounding box regression loss with dynamic focusing mechanism[J]. arXiv preprint arXiv:2301.10051, 2023.
5. Stein G P, Mano O, Shashua A. Vision-based ACC with a single camera: bounds on range and range rate accuracy[C]//IEEE IV2003 intelligent vehicles symposium. Proceedings (Cat. No. 03TH8683). IEEE, 2003: 120-125.