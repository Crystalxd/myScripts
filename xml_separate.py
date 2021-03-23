import os, sys
import argparse
import os.path as op
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
class Annotation():
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.size = self.root.find('size')
        self.width = int(self.size.find('width').text)
        self.height = int(self.size.find('height').text)

        # self.labelLists = self.LabelLists()

    def LabelRects(self):       # 返回xml里所有目标的值 [['name', xmin, ymin, xmax, ymax], ['name', xmin, ymin, xmax, ymax], ...]
        labelRects = []
        objects = self.root.findall('object')
        for object in objects:
            label = object.find('name').text
            bndbox = object.findall('bndbox')
            for box in bndbox:
                xmin = box.find('xmin').text
                ymin = box.find('ymin').text
                xmax = box.find('xmax').text
                ymax = box.find('ymax').text
                labelRects.append([label, xmin, ymin, xmax, ymax])
        if labelRects == []:
            return None
        else:
            return labelRects

    def removeNullObjectFile(self):    # 移除没有目标框的xml文件
        objects = self.root.findall('object')
        if objects == []:
            os.remove(self.path)


    def separateAllDiffObject(self, label, dstPath):
        objects = self.root.findall('object')
        if objects == []:
            print("No object to remove.")
            return 0
        else:
            for object in objects:
                className = object.find('name').text
                if className != label:
                    self.root.remove(object)
            if self.root.findall('object') != []:
                self.tree.write(dstPath)

    def showAllLabel(self, classList=None):     # 显示xml文件中所有类别
        allLabel = []
        if self.LabelRects() != None:
            for object in self.LabelRects():
                if object[0] not in allLabel:
                    allLabel.append(object[0])
        return allLabel

def getAllXmlsLabel(xmlPath):
    if os.path.isfile(xmlPath):
        print('This str is not a directory.')
    else:
        allLabel = []
        for xmlfile in os.listdir(xmlPath):
            xml = Annotation(os.path.join(xmlPath, xmlfile))
            if xml.LabelRects() != None:
                for object in xml.LabelRects():
                    if object[0] not in allLabel:
                        allLabel.append(object[0])
        return allLabel


def parse_args():
    parser = argparse.ArgumentParser(description='xmlfiles process')
    # parser.add_argument('--type', type=str, default='separateObjectNode', help='separate object to single xml.')
    parser.add_argument('--source', '-s', type=str, default='', help='source directory')
    parser.add_argument('--dst', type=str, default='', help='the target directory')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # args = parse_args()
    xmlPath = r"D:\Liuxx\Tests\3" # args.source # r"D:\liuxx\Datasets\压板\标注"    # 5741575967273_pic_hd.xml   test_none
    dstPath = r"D:\Liuxx\Tests\5" # args.dst # r"D:\liuxx\Datasets\压板\test"
    os.makedirs(dstPath, exist_ok=True)

    if os.path.isfile(xmlPath):
        xml = Annotation(xmlPath)
        print(xml.LabelRects())
    else:
        allLabel = getAllXmlsLabel(xmlPath)
        print(allLabel)
        for label in allLabel:
            newXmlPath = os.path.join(dstPath, label)
            os.makedirs(newXmlPath, exist_ok=True)
            for xmlfile in tqdm(sorted(os.listdir(xmlPath))):
                xml = Annotation(os.path.join(xmlPath, xmlfile))
                xml.separateAllDiffObject(label, os.path.join(newXmlPath, xmlfile))
                # newxml = Annotation(os.path.join(newXmlPath, xmlfile))
                # newxml.removeNullObjectFile()
