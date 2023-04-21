import argparse
import os
import platform
import sys
from pathlib import Path
import pyautogui
import torch
from screeninfo import get_monitors
import re
import xml.etree.ElementTree as ET

# 获取该文件的绝对路径，输出为：B:\BaiduNetdiskDownload\ubuntu 实验\yolov5\detect.py
FILE = Path(__file__).resolve()
# 获取该文件的绝对路径，输出为：B:\BaiduNetdiskDownload\ubuntu 实验\yolov5\detect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory
# 获取该文件的绝对路径，输出为：B:\BaiduNetdiskDownload\ubuntu 实验\yolov5\detect.py
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# 将其绝对路径转换为相对路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 加载完模型之后，对应读取模型的步长、类别名、pytorch模型类型
    stride, names, pt = model.stride, model.names, model.pt
    # 判断模型步长是否为32的倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # 屏幕截图，加载带预测图
    # source: [screen, left, top, width, height]
    # dataset = _, img, original img, _, info
    dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    # 通过运行一次推理来预热模型（内部初始化一张空白图预热模型）
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Directories
    save_dir = increment_path("./runs/detect")  # increment run

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # dataset数据集遍历，path为图片路径
    # im为压缩后的图片， 640 * 480 * 3
    # im0s为原图，1080 * 810
    # vid_cap 空
    # s 打印图片的信息
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # numpy array to tensor and device
            # 在模型中运算，需要转换成pytorch，从numpy转成pytorch
            # 在将其数据放入到cpu 或者 gpu中
            im = torch.from_numpy(im).to(model.device)
            # 半精度训练 uint8 to fp16/32
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 归一化
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 如果图片为3维，则增加一个维度，即为batch_size
            if len(im.shape) == 3:
                # 图片为3维(RGB)，在前面添加一个维度，batch_size=1。本身输入网络的图片需要是4维， [batch_size, channel, w, h]
                # 【1，3，640，480】
                im = im[None]  # expand for batch dim

        # Inference
        # visualize 一开始为false，如果为true则对应会保存一些特征
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 数据的推断增强，但也会降低速度。最后检测出的结果为18900个框
            # 结果为【1，18900，85】，预训练有85个预测信息，4个坐标 + 1个置信度 +80各类别
            pred = model(im, augment=augment, visualize=visualize)

        # NMS 非极大值阈值过滤
        # conf_thres: 置信度阈值；iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None
        # agnostic_nms: 进行nms是否也去除不同类别之间的框 默认False
        # max_det: 每张图片的最大目标个数 默认1000，超过1000就会过滤
        # pred: [1,num_obj, 6] = [1,5, 6]   这里的预测信息pred还是相对于 img_size(640) 。本身一开始18900变为了5个框，6为每个框的 x左右y左右 以及 置信度 类别值
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        # 对每张图片进行处理，将pred(相对img_size 640)映射回原图img0 size
        # 此处的det 表示5个检测框中的信息
        for i, det in enumerate(pred):  # per image
            # 每处理一张图片，就会加1
            seen += 1
            # p为当前图片或者视频绝对路径
            # im0原始图片
            # frame: 初始为0  可能是当前图片属于视频中的第几帧
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            # gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # 绘图工具，画图检测框的粗细，一种通过PIL，一种通过CV2处理
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息（相对img_size 640）映射回原图 img0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 统计每个框的类别
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    # -----------------------------------------------------------------------
                    # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    get_position_action(xywh)
                    # -----------------------------------------------------------------------

                    # 在原图上画框 + 将预测到的目标剪切出来
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            # 通过imshow显示出框
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # seen为预测图片总数，dt为耗时时间，求出平均时间
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


# object position function
def get_position_action(data):
    pass


# get windows screen number
def get_window_screen_number(window_title):
    for monitor in get_monitors():
        # Get the position of the top-left corner of the monitor
        monitor_x, monitor_y = monitor.x, monitor.y
        # Get the width and height of the monitor
        monitor_width, monitor_height = monitor.width, monitor.height

        for win in pyautogui.getAllWindows():
            if win.title == window_title:
                left, top = win.topleft
                if monitor_x <= left <= monitor_x + monitor_width and monitor_y <= top <= monitor_y + monitor_height:
                    return re.findall(r'\d+', monitor.name)[0]
    return None


if __name__ == '__main__':
    # 获取所有窗口
    windows_list = pyautogui.getAllWindows()
    print("当前所有窗口：")
    # 遍历所有窗口
    windows = []
    for window in windows_list:
        if window.title != "":
            windows.append(window.title)
    for i, window in enumerate(windows):
        print("%d %s" % (i, window))
    print("--------------------------------------------------")
    title = windows[int(input("请输入窗口标题："))]
    print("窗口标题：", title)
    # 通过窗口标题获取窗口
    window = pyautogui.getWindowsWithTitle(title.strip())[0]
    screen_number = get_window_screen_number(title.strip())
    x, y, w, h = window.left, window.top, window.width, window.height
    print("屏幕编号：", screen_number)
    print("窗口位置：", x, y, w, h)
    print("--------------------------------------------------")
    print("加载模型...")
    source = "%s %s %s %s %s" % (screen_number, x, y, w, h)

    tree = ET.parse('./parameter.xml')
    root = tree.getroot()
    element = {}
    for elem in root:
        tag = elem.tag
        attrib = elem.attrib['attribute']
        # 转换数据类型
        if attrib[0] == '[' and attrib[-1] == ']':
            attrib = list(map(int, attrib[1:-1].split(',')))
        elif attrib.isdigit():
            attrib = int(attrib)
        elif attrib.replace('.', '', 1).isdigit():
            attrib = float(attrib)
        elif attrib == 'True':
            attrib = True
        elif attrib == 'False':
            attrib = False
        element[tag] = attrib

    data = "./data/coco128.yaml"
    device = element['device']
    weights = element['weights']
    imgsz = element['img-size']
    conf_thres = element['conf_thres']
    iou_thresh = element['iou_thresh']
    max_det = element['max_det']
    classes = element['classes']
    augment = element['augment']
    visualize = element['visualize']
    line_thickness = element['line_thickness']
    half = element['half']
    dnn = element['dnn']
    # 开始
    run(weights, source, data, imgsz, conf_thres, iou_thresh, max_det, device, classes,
        False, augment, visualize, line_thickness, False, False, half, dnn)