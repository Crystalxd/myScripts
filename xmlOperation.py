import os, shutil, cv2
import argparse
import sys
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

    def modifyCoordinate(self):
        objects = self.root.findall('object')
        for object in objects:
            bndbox = object.findall('bndbox')
            for box in bndbox:
                box.find('xmin').text = roundCoord(box.find('xmin').text)
                box.find('ymin').text = roundCoord(box.find('ymin').text)
                box.find('xmax').text = roundCoord(box.find('xmax').text)
                box.find('ymax').text = roundCoord(box.find('ymax').text)
                self.tree.write(self.path)

    def modifyName(self, dstPath):
        objects = self.root.findall('object')
        if objects != []:
            for object in objects:
                className = object.find('name').text
                if className == 'lineboard':
                    # print(self.name)
                    object.find('name').text = 'signboard'
                if className == 'person':
                    object.find('name').text = 'nohelmet'
                self.tree.write(dstPath)

    def removeObject(self, classList, dstPath):        # 移除对应类别的节点
        objects = self.root.findall('object')
        if objects == []:
            print("No object to remove.")
        else:
            for object in objects:
                className = object.find('name').text
                if className in classList:
                    self.root.remove(object)
                self.tree.write(dstPath)


    def removeNullObjectFile(self, dstPath):    # 移除没有目标框的xml文件
        objects = self.root.findall('object')
        if objects == []:
            # os.remove(self.path)
            # print(self.name)
            shutil.move(self.path, dstPath)

    def separateObjectNode(self, classList, path):       # 分离xml指定类别目标
        # if self.LabelRects() == None:
        #     print(self.name + " has no object.")
        # else:
        #     objects = self.LabelRects()
        #     for object in objects:
        #         if object[0] == "red":
        #             print(self.name)
        sourceFiles = []
        objects = self.root.findall('object')
        if objects == []:
            print("No object to remove.")
            return 0
        else:
            for object in objects:
                className = object.find('name').text
                if className in classList:
                    self.root.remove(object)
                    if self.name not in sourceFiles:
                        sourceFiles.append(self.name)
                    self.tree.write(path)
            return sourceFiles

    def separateAllDefObject(self, label, dstPath):
        objects = self.root.findall('object')
        if objects == []:
            print("No object to remove.")
            return 0
        else:
            for object in objects:
                className = object.find('name').text
                if className != label:
                    self.root.remove(object)
                self.tree.write(dstPath)

    def showAllLabel(self, classList=None):     # 显示xml文件中所有类别
        allLabel = []
        if self.LabelRects() != None:
            for object in self.LabelRects():
                if object[0] not in allLabel:
                    allLabel.append(object[0])
        return allLabel

    def rmXmlByObject(self, imgPath, dstXmlPath, dstImgPath):
        objects = self.root.findall('object')
        for object in objects:
            if object.find('name').text == 'drop':
                shutil.move(self.path, dstXmlPath)
                imgName = self.name[:-4] + '.jpg'
                shutil.move(op.join(imgPath, imgName), dstImgPath)

def roundCoord(coord):
    return str(round(float(coord)))

def getAllXmlsLabel(xmlPath):
    if os.path.isfile(xmlPath):
        print('This str is not a directory.')
    else:
        for xmlfile in os.listdir(xmlPath):
            xml = Annotation(os.path.join(xmlPath, xmlfile))
            allLable = xml.showAllLabel()
        return allLable


def moveImgsFromXml(path, savePath, xmlList):
    imgs = os.listdir(path)
    os.makedirs(savePath, exist_ok=True)
    for img in tqdm(imgs):
        if img[:-4] in xmlList:
            shutil.move(os.path.join(path, img), savePath)

def cv_imread(image_path):
    cv_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return cv_img

def parse_args():
    parser = argparse.ArgumentParser(prog='xmlfiles process')
    # parser.add_argument('--type', type=str, default='separateObjectNode', help='separate object to single xml.')
    parser.add_argument('--source', '-s', type=str, default='', help='source directory')
    parser.add_argument('--dst', type=str, default='', help='the target directory')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    xmlPath = r"D:\Liuxx\Datasets\手机拍照行为\TJPZ20200708done_xml"
    # xmlPath = r"D:\Liuxx\Datasets\玩手机\xml"    # 5741575967273_pic_hd.xml   test_none
    # xmlPath = r"D:\Liuxx\Tests\yolo4预标注\xml"
    dstPath = r"D:\Datasets\玩手机\dropout"
    # os.makedirs(dstPath, exist_ok=True)
    '''
    picPath = r"D:\Datasets\zhongmai\imgs"
    
    '''
    # args = parse_args()
    # savePath = "D:"+os.sep+"Datasets"+os.sep+"烟火标注数据集_finally"+os.sep+"xml_save"
    # imgPath = "D:"+os.sep+"Datasets"+os.sep+"烟火标注数据集_finally"+os.sep+"pic"
    # dstImgPath = "D:"+os.sep+"Datasets"+os.sep+"VOCdevkit"+os.sep+"VOC2012"+os.sep+"JPEGImages_person"
    # saveRmXml = r"D:\Datasets\人体安全帽样本\helmet_part_0715\2\removeXml"
    if os.path.isfile(xmlPath):
        xml = Annotation(xmlPath)
        print(xml.LabelRects())
        # xml.separateObjectNode()
        # xml.removeObject()
        # xml.showAllLabel()
    else:
        # os.makedirs(savePath, exist_ok=True)
        # classList = ['light']
        # xmlList = []
        # # allLabel = getAllXmlsLabel(xmlPath)
        # print(allLabel)
        # for label in allLabel:
            # dstPath = os.path.join(dstPath, label)
        os.makedirs(dstPath, exist_ok=True)
        for xmlfile in tqdm(sorted(os.listdir(xmlPath))):
            xml = Annotation(os.path.join(xmlPath, xmlfile))

            xml.modifyCoordinate()
            # xml.modifyName(op.join(dstPath, xml.name))
            # xml.removeNullObjectFile(dstPath)
            # xml.rmXmlByObject(imgPath=r"D:\Datasets\zhongmai\imgs", dstXmlPath=r"D:\Datasets\zhongmai\dropout"
            #                                                                   r"", dstImgPath=r"D:\Datasets\zhongmai\dropout\img")
            # xml.modifyName(os.path.join(xmlPath, xmlfile))
            # xml.removeObject(['person'], os.path.join(dstPath, xmlfile))
            # xml.separateAllDefObject(label, os.path.join(dstPath, xmlfile))
            # separateAllDefObject

    #except Exception as e:
    #    print(xmlfile, e)

        """
        os.makedirs(saveRmXml, exist_ok=True)
        start_num = len(os.listdir(xmlPath))
        for xmlfile in sorted(os.listdir(xmlPath)):
            # print(xmlfile)
            try:
                xml = Annotation(os.path.join(xmlPath, xmlfile))
                xml.removeNullObjectFile(saveRmXml)
            except Exception as e:
                print(e)
                print(xmlfile)
                # shutil.move(op(xmlPath, xmlfile), saveRmXml)
            '''
            try:
                xml = Annotation(os.path.join(xmlPath, xmlfile))
                # xml.removeNullObjectFile(rmPath)
                for labelRect in xml.LabelRects():
                    if LabelRect[0] == 'light':
                        pass
            except:
                print(xmlfile + "is not a valid annotation file!")
                shutil.move(os.path.join(xmlPath, xmlfile), rmPath)
            '''

            #xmlList.append(xml.name[:-4])
        #moveImgsFromXml(imgPath, dstImgPath, xmlList)

            #sourceFiles = xml.separateObjectNode(classList, os.path.join(savePath, xmlfile))
        #print(sourceFiles)
            # if xml.width == 0 or xml.height == 0:
            #     print(xml.name)
                #imgName = xml.name[:-4] + '.jpg'
                #img = cv2.imread(os.path.join(imgPath, imgName))
                #print(os.path.join(imgPath, imgName))
                #print(xml.name + " : {}".format(img))


        end_num = len(os.listdir(xmlPath))
        print("移除的xml文件数目：%d" % (start_num - end_num))
        if os.path.getsize(saveRmXml) == 0:
            os.removedirs(saveRmXml)
        # xml = Annotation(xmlPath)
        # print(xml.name)
        # xml.remove_object()
        """