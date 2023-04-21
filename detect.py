import argparse
import os
import platform
import sys
from pathlib import Path

import torch

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


# 该注解是自个定义的注解，主要的功能是判断torch版本
# 如果torch>=1.9.0则应用torch.inference_mode()装饰器，否则使用torch.no_grad()装饰器
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # 此行代码将其source 转换字符串
    # source 为命令行传入的图片或者视频，大致为：python detect.py --source data/images/bus.jpg
    source = str(source)
    # 是否保存预测后的图片
    # source传入的参数为jpg而不是txt，为true
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Path(source)：为文件地址，data/images/bus.jpg
    # suffix[1:]：截取文件后缀，即为bus.jpg，而[1:]则为jpg后，最后输出为jpg
    # 判断该jpg是否在(IMG_FORMATS + VID_FORMATS) 该列表内，该列表可参照下一个代码模块。最后输出为true
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断是否为网络流地址或者是网络的图片地址
    # 将其地址转换为小写，并且判断开头是否包括如下网络流开头的
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 是否是使用webcam 网页数据，一般为false
    # 判断source是否为数值（0为摄像头路径）或者txt文件 或者 网络流并且不是文件地址
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # 是否传入的为屏幕快照文件
    screenshot = source.lower().startswith('screen')
    # 如果是网络流地址 以及文件，则对应下载该文件
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 加载完模型之后，对应读取模型的步长、类别名、pytorch模型类型
    stride, names, pt = model.stride, model.names, model.pt
    # 判断模型步长是否为32的倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # 流加载器，类似代码 python detect.py——source 'rtsp://example.com/media.mp4'
    # 命令行此处定义的webcam 为flase，所以跳过该逻辑
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        # 屏幕截图，加载带预测图
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 加载图
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # 通过运行一次推理来预热模型（内部初始化一张空白图预热模型）
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

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

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 对每张图片进行处理，将pred(相对img_size 640)映射回原图img0 size
        # 此处的det 表示5个检测框中的信息
        for i, det in enumerate(pred):  # per image
            # 每处理一张图片，就会加1
            seen += 1
            # 输入源是网页，对应取出dataset中的一张照片
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                # p为当前图片或者视频绝对路径
                # im0原始图片
                # frame: 初始为0  可能是当前图片属于视频中的第几帧
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 图片的保存路径
            save_path = str(save_dir / p.name)  # im.jpg
            # txt 保存路径（保存预测框的坐标）
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            # txt 保存路径（保存预测框的坐标）
            s += '%gx%g ' % im.shape[2:]  # print string
            # gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc: for save_crop 在save_crop中使用
            imc = im0.copy() if save_crop else im0  # for save_crop

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
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
                    if save_txt:  # Write to file
                        # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # 如果需要就将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 通过imshow显示出框
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # seen为预测图片总数，dt为耗时时间，求出平均时间
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    # 保存预测的label信息 xywh等   save_txt
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # strip_optimizer函数将optimizer从ckpt中删除  更新模型
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    # 传入的参数，以上的参数为命令行赋值的参数，如果没有给定该参数值，会有一个default的默认值进行赋值
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()

    # 此处对传入的size加以判断。如果不传入，默认为640，则长度为1，则对应size 为640 * 640。如果传入的参数为640 * 640 ，则不修改
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 将其所有的参数信息进行打印
    # print_args(vars(opt))
    return opt


def main(opt):
    # 检查requirement的依赖包 有无成功安装，如果没有安装部分会在此处报错
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
