# Object_Detection
Object_Detection项目克隆自[ultralytics/yolov5](https://github.com/ultralytics/yolov5)，将其修改使项目能够应用于电脑应用程序
## 安装

克隆项目

```bash
git clone git@github.com:Karenina-na/Object_Detection.git
```

```shell
pip install -r requirements.txt
```

```shell
python main.py
```
## 使用方法

- parameter.xml 为模型推理参数列表，其中检测类别的标签值类可在/data/下进行选择。
- device默认为0，可根据电脑状态更改为cpu或其他数值。



## 作者

- [Ultralytics](https://github.com/ultralytics)
- [Karenina-na](https://github.com/Karenina-na) (me)
## 证书

[AGPL-3.0 License](https://github.com/ultralytics/yolov5/blob/master/LICENSE) from Ultralytics \
[MIT License](https://choosealicense.com/licenses/mit/) from Karenina-na
## Yolo v5 简介
YOLOv5 🚀 是世界上最受欢迎的视觉 AI，代表<a href="https://ultralytics.com"> Ultralytics </a>对未来视觉 AI 方法的开源研究，结合在数千小时的研究和开发中积累的经验教训和最佳实践。

如果要申请企业许可证，请填写表格<a href="https://ultralytics.com/license">Ultralytics 许可</a>.
## 预训练模型

### 预训练模型

| 模型                                                                                             | 尺寸<br><sup>（像素） | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | 推理速度<br><sup>CPU b1<br>（ms） | 推理速度<br><sup>V100 b1<br>（ms） | 速度<br><sup>V100 b32<br>（ms） | 参数量<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|------------------------------------------------------------------------------------------------|-----------------|----------------------|-------------------|-----------------------------|------------------------------|-----------------------------|-----------------|------------------------|
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)             | 640             | 28.0                 | 45.7              | **45**                      | **6.3**                      | **0.6**                     | **1.9**         | **4.5**                |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)             | 640             | 37.4                 | 56.8              | 98                          | 6.4                          | 0.9                         | 7.2             | 16.5                   |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)             | 640             | 45.4                 | 64.1              | 224                         | 8.2                          | 1.7                         | 21.2            | 49.0                   |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)             | 640             | 49.0                 | 67.3              | 430                         | 10.1                         | 2.7                         | 46.5            | 109.1                  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)             | 640             | 50.7                 | 68.9              | 766                         | 12.1                         | 4.8                         | 86.7            | 205.7                  |
|                                                                                                |                 |                      |                   |                             |                              |                             |                 |                        |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)           | 1280            | 36.0                 | 54.4              | 153                         | 8.1                          | 2.1                         | 3.2             | 4.6                    |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt)           | 1280            | 44.8                 | 63.7              | 385                         | 8.2                          | 3.6                         | 12.6            | 16.8                   |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt)           | 1280            | 51.3                 | 69.3              | 887                         | 11.1                         | 6.8                         | 35.7            | 50.0                   |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt)           | 1280            | 53.7                 | 71.3              | 1784                        | 15.8                         | 10.5                        | 76.8            | 111.4                  |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt)<br>+[TTA] | 1280<br>1536    | 55.0<br>**55.8**     | 72.7<br>**72.7**  | 3136<br>-                   | 26.2<br>-                    | 19.4<br>-                   | 140.7<br>-      | 209.8<br>-             |
