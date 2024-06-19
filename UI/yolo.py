import sys
import os

# 添加 ultralytics 目录的父目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class yolo(object):
    def __init__(self, signal, weights_path='../model/best.pt', confidence=0.3):
        """
                初始化YOLOv8对象。

                参数：
                weights_path: str，YOLOv8权重文件路径
                confidence: float，置信度阈值
        """
        # 加载YOLOv8模型
        self.model = YOLO(weights_path,task='dectect')

        # 加载信号
        self.signal = signal

        # 初始化对象属性
        self.confidence = confidence
        self.class_names = self.model.model.names
        self.class_names2id = {}
        for i, name in self.class_names.items():
            self.class_names2id[name] = i

    def detect_image(self, opencv_img, show_log=False):
        """
               对图像进行目标检测，返回检测结果。

               参数：
               img: numpy array，待检测图像
               show_log: bool，是否显示检测日志

               返回：
               result_list: list，检测结果，每个元素是一个列表，格式为：
               [类别名称, 置信度, xmin, ymin, xmax, ymax]
        """
        # 对图像进行目标检测
        results = self.model(opencv_img, verbose=show_log)
        # 获取检测结果
        bbox = results[0].boxes
        # 初始化检测结果列表
        result_list = []
        # 遍历检测结果
        for idx in range(len(bbox.data)):
            # 获取检测框的坐标和置信度等信息
            xmin = int(bbox.data[idx][0])  # 检测框左上角 x 坐标
            ymin = int(bbox.data[idx][1])  # 检测框左上角 y 坐标
            xmax = int(bbox.data[idx][2])  # 检测框右下角 x 坐标
            ymax = int(bbox.data[idx][3])  # 检测框右下角 y 坐标
            conf = round(float(bbox.data[idx][4]), 2)  # 检测框置信度
            cls_idx = int(bbox.data[idx][5])  # 检测框对应的类别索引
            # 如果置信度大于等于阈值，则将检测结果加入结果列表
            if conf >= self.confidence:
                result_list.append([self.class_names[cls_idx], conf, xmin, ymin, xmax, ymax])
        return result_list  # [[name,conf,xmin,ymin,xmax,ymax]]

    def calculate_distance(self, object_info):
        """
        根据物体在图像中的位置计算其到摄像头的距离。
        :param object_info: 包含物体类别和坐标的列表，格式为[[name,conf,xmin,ymin,xmax,ymax]]
        :return: 物体到摄像头的距离，单位为米。
        """
        _,_,x_min, y_min, x_max, y_max = object_info
        object_width = x_max - x_min
        object_height = y_max - y_min
        focal_length = 2.5  # 摄像头的焦距，单位为毫米
        sensor_width = 3.68  # 摄像头的感光元件尺寸，单位为毫米
        image_width = 1920  # 图像的宽度，单位为像素
        image_height = 1080  # 图像的高度，单位为像素
        pixel_size = sensor_width / image_width  # 像素尺寸，单位为毫米
        object_width_mm = object_width * pixel_size  # 物体在图像中的宽度，单位为毫米
        object_height_mm = object_height * pixel_size  # 物体在图像中的高度，单位为毫米
        object_width_pix = object_width_mm * focal_length / 1000 / pixel_size  # 物体在实际距离上的宽度，单位为像素
        object_height_pix = object_height_mm * focal_length / 1000 / pixel_size  # 物体在实际距离上的高度，单位为像素
        image_width_mm = image_width * pixel_size  # 图像的宽度，单位为毫米
        image_height_mm = image_height * pixel_size  # 图像的高度，单位为毫米
        image_diagonal_mm = math.sqrt(image_width_mm ** 2 + image_height_mm ** 2)  # 图像对角线的长度，单位为毫米
        object_diagonal_pix = math.sqrt(object_width_pix ** 2 + object_height_pix ** 2)  # 物体对角线的长度，单位为像素
        object_diagonal_mm = object_diagonal_pix * pixel_size  # 物体对角线的长度，单位为毫米
        dis = (focal_length * object_diagonal_mm) / math.sqrt(
            (object_diagonal_pix ** 2) + (image_diagonal_mm ** 2))
        distance = min(1920 / object_width, 1360 / object_height)+dis
        return round(distance, 2)

    def draw_annotated_image(self, detection_results, image):
        """
            在给定的OpenCV图像上绘制检测结果并标注物体距离。
            :param detection_results: 检测结果列表，每个元素为一个包含物体类别和坐标信息的列表。
            :param opencv_img: OpenCV图像，可以是图片或者视频帧。
            :return: 绘制了检测结果和标注距离的OpenCV图像。
        """
        # 初始化 Annotator 类的实例，指定线条宽度和字体大小
        annotator = Annotator(image, line_width=2, font_size=2)
        # 遍历所有检测结果
        for result in detection_results:
            # 计算物体到摄像头的距离
            distance = self.calculate_distance(result)

            # 如果检测结果不是车辆，则使用类别名和距离标注边框
            if result[0] != "car" and result[0] != "truck":
                label_text = result[0] + ',' + str(distance) + "m"
                # 选择颜色
                color_id = self.class_names2id[result[0]]
                # 标注边框
                annotator.box_label(result[2:6], label_text, color=colors(color_id, True))
            else:
                label_text = result[0] + ',' + str(distance) + "m"
                # 如果车辆距离小于 10 米，则使用红色标注边框
                if distance < 10:
                    annotator.box_label(result[2:6], label_text, color=colors(0, True))
                # 如果车辆距离在 10~20 米之间，则使用橙色标注边框
                elif distance < 20:
                    annotator.box_label(result[2:6], label_text, color=colors(2, True))
                # 如果车辆距离大于 20 米，则使用绿色标注边框
                else:
                    annotator.box_label(result[2:6], label_text, color=colors(8, True))
        # 返回标注过的图像
        return annotator.im

    def imshow(self, result_list, opencv_img):
        """
        在图像上绘制检测结果，并显示图像。

        :param result_list: 检测结果列表，包含检测到的物体的类别、位置等信息。
        :param opencv_img: 要显示的图像。
        """
        if len(result_list) > 0:
            annotated_img = self.draw_annotated_image(result_list, opencv_img)
            # 如果有检测结果，则在原图上绘制标注后的图像
        else:
            annotated_img = opencv_img
            # 如果没有检测结果，直接使用原图
        self.signal.emit(opencv_img)
        # 显示图像，并等待按键响应

    def detect_video(self, video_file):
        """
        对视频与摄像头进行目标检测，返回检测结果。
        :param video_file:
        :return:
        """

        # 打开视频文件
        cap = cv2.VideoCapture(video_file)
        # 获取视频帧率，宽度和高度
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},width={},height={}".format(frame_fps, frame_width, frame_height))

        while True:
            # 读取一帧视频
            ret, frame = cap.read()
            if not ret:
                break
            # 对视频帧进行目标检测
            result_list = self.detect_image(frame)
            # 在帧上绘制检测结果并显示
            result_img = self.draw_annotated_image(result_list, frame)
            self.signal.emit(result_img)

        # 关闭视频文件并销毁所有窗口
        cap.release()


if __name__ == '__main__':
    detector = yolo()
    detector.detect_video(r'D:\car.mp4')
