import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
from tqdm import tqdm

# 使用python3.7版本执行此脚本

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection Label Tools")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--show", action='store_false',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in voc format")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.35,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape  # cfg模型输入大小：608x608
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            print(bbox)
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def writexml(fileDir, imgInfo, rectLists):
    f = open(fileDir, "w", encoding='UTF-8')
    f.write("<annotation>\n")
    f.write("  <folder>train_images</folder>\n")
    f.write("  <filename>{}</filename>\n".format(os.path.basename(imgInfo[0])))
    f.write("  <path>{}</path>\n".format(imgInfo[0]))
    f.write("  <source>\n")
    f.write("      <database>Unknow</database>\n")
    f.write("  </source>\n")
    f.write("  <size>\n")
    f.write("      <width>%d</width>\n" % imgInfo[2])
    f.write("      <height>%d</height>\n" % imgInfo[1])
    f.write("      <depth>{}</depth>\n".format(imgInfo[3]))
    f.write("  <segmented>0</segmented>\n")
    f.write("  </size>\n")

    for rectList in rectLists:
        f.write("  <object>\n")
        f.write("      <name>{}</name>\n".format(rectList[0]))
        f.write("      <pose>0</pose>\n")
        f.write("      <truncated>0</truncated>\n")
        f.write("      <difficult>0</difficult>\n")
        f.write("      <bndbox>\n")
        f.write("          <xmin>%d</xmin>\n" % rectList[1])
        f.write("          <ymin>%d</ymin>\n" % rectList[2])
        f.write("          <xmax>%d</xmax>\n" % rectList[3])
        f.write("          <ymax>%d</ymax>\n" % rectList[4])
        f.write("      </bndbox>\n")
        f.write("  </object>\n")
    f.write("</annotation>\n")
    f.close()

def getImgInfo(path):
    img = cv2.imread(path)
    height, width, depth = img.shape
    imgInfo = [os.path.abspath(path), height, width, depth]
    return imgInfo

def save_xml(name, image, detections, xmlDir, labels):
    """
    Files saved with image_name.xml and relative coordinates
    """
    file_name = os.path.join(xmlDir, os.path.basename(name).split('.')[0] + '.xml')     # 待写入的xml文件全路径
    imgInfo = getImgInfo(name)           # 获取原图片信息：[绝对路径，高度，宽度，通道数]
    im_height, im_width = imgInfo[1:-1]  # 原图片的尺寸
    rectLists = []      # 用于存放所有预测框
    for label, confidence, bbox in detections:
        x, y, w, h = convert2relative(image, bbox)         # 在608x608(模型输入尺寸)图片上的中心点坐标(x, y)以及框宽与框高大小（比例化数值）
        # x, y, w, h = x * im_width, y * im_height, w * im_width, h * im_height            # 在原图片上的中心点坐标(x, y)以及框宽与框高大小
        xmin = max(float(x) - float(w) / 2, 0)
        xmax = min(float(x) + float(w) / 2, 1)
        ymin = max(float(y) - float(h) / 2, 0)
        ymax = min(float(y) + float(h) / 2, 1)

        xmin = int(im_width * xmin)
        xmax = int(im_width * xmax)
        ymin = int(im_height * ymin)
        ymax = int(im_height * ymax)
        # # 将yolo格式的预测框坐标转换成voc格式所需要的值（左上点坐标，右下点坐标）
        # xmin = int(x - w / 2)
        # ymin = int(y - h / 2)
        # xmax = int(x + w / 2)
        # ymax = int(y + h / 2)
        # # 在转换后点的坐标可能会越界（超出图片范围），因此对于越界的坐标需要调整
        # if xmin < 0:
        #     xmin = 1
        # if xmax > im_width:
        #     xmax = im_width
        # if ymin < 0:
        #     ymin = 1
        # if ymax > im_height:
        #     ymax = im_height


        rectList = [label, xmin, ymin, xmax, ymax]    # 一帧图中每个预测框的属性: [类别，左上点横坐标，左上点纵坐标，右下点横坐标，右下点横坐标]
        # 此处if主要是对预测标签的提取，如果labels制定了标签，则按标签处理；若labels = []，则处理所有标签
        if labels != []:
            if label in labels:
                rectLists.append(rectList)
        else:
            rectLists.append(rectList)
    if rectLists != []:
        writexml(file_name, imgInfo, rectLists)
    else:
        print("No objects were detected!\n")


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def main(labels):
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    images = load_images(args.input)

    if args.save_labels:
        # xmlDir = "../../darknet/preLabel/xml"
        xmlDir = input("Please input saved path of xml: ")
        os.makedirs(xmlDir, exist_ok=True)
    for image_name in tqdm(sorted(images)):
        try:
            image, detections = image_detection(
                image_name, network, class_names, class_colors, args.thresh
                )
        except:
            print(image_name + ' cannot read, skip!\n')
            continue
        if args.save_labels:
            # save_annotations(image_name, image, detections, class_names)
            save_xml(image_name, image, detections, xmlDir, labels)
        # darknet.print_detections(detections, args.ext_output)
        # fps = int(1/(time.time() - prev_time))
        # print("FPS: {}".format(fps))
        if not args.show:
            cv2.imshow('Inference', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break



if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    # 先将该脚本复制到darknet根目录下
    # 运行命令：python darknet_label.py --input images --weights yolov4.weights --save_labels

    # labels指定写入xml目标的类别，若labels = [],则写入所有类别
    labels = ['person']

    main(labels)
