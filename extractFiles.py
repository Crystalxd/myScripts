"""
Author: liuxx
此脚本用于从一个文件夹中抽取指定数量的文件，包括随机抽取方式和顺序抽取方式
"""

import os, shutil, random
import os.path as osp

def extFiles(use_type, count, srcPath, dstPath): # srcPath为需要移动文件的目录，dstPath为移动文件的目标目录
	TYPE = ['sorted', 'random']
	extType = TYPE[use_type]
	if not os.path.exists(dstPath):
		os.mkdir(dstPath)
	if extType == 'sorted': # 将文件名排序选取
		files = sorted(os.listdir(srcPath))
		#fileNum = len(files)
		slice = files[0:count] # 取多少个样本
		for file in slice:
			shutil.move(osp.join(srcPath, file), dstPath)
	elif extType == 'random': # 随机选取
		sample = random.sample((os.listdir(srcPath)), count)
		for file in sample:
			shutil.move(osp.join(srcPath, file), dstPath)

# 分离pic和json文件
def move_json_from_pics(pics, jsonDir):
	if not osp.exists(jsonDir):
		os.mkdir(jsonDir)
	for file in os.listdir(pics):
		if file.split('.')[1] == 'json':
			shutil.move(osp.join(pics, file), jsonDir)

def movePicAccordingXml(picPath, xmlPath, savePath):
    os.makedirs(savePath, exist_ok=True)
    xmlList = sorted(os.listdir(xmlPath))
    for pic in os.listdir(picPath):
       if pic[:-4]+'.xml' in xmlList:
           shutil.move(os.path.join(picPath, pic), savePath)

def moveXmlAccordingPic(xmlPath, picPath, savePath):
	os.makedirs(savePath, exist_ok=True)
	picList = sorted(os.listdir(picPath))
	for xml in os.listdir(xmlPath):
		if xml[:-4]+'.jpg' in picList:
			shutil.move(os.path.join(xmlPath, xml), savePath)


if __name__ == '__main__':
	# srcPath = "/home/dl/liuxx/yolov5_for_rknn/person_helmet/images/train2021"   # 源路径
	# dstPath = "/home/dl/liuxx/yolov5_for_rknn/person_helmet/images/val2021" # 目标路径
	# count = 7430  	 # 样本数
	# use_type = 1     # 0 表示连续抽取, 1 表示随机抽取
	# extFiles(use_type, count, srcPath, dstPath)

    picPath = "/home/dl/liuxx/yolov5_for_rknn/person_helmet/images/val2021"
    labelPath = "/home/dl/liuxx/yolov5_for_rknn/person_helmet/labels/train2021"
    savePath = "/home/dl/liuxx/yolov5_for_rknn/person_helmet/labels/val2021"
    movePicAccordingXml(labelPath, picPath, savePath)
    print("Done.")
