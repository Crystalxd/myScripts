import os, sys
import shutil
import xml.etree.ElementTree as ET


def megerXml(xml_1, xml_2):
    tree_1 = ET.parse(xml_1)
    root_1 = tree_1.getroot()
    new = tree_1
    if xml_2:
        tree_2 = ET.parse(xml_2)
    # root_2 = tree_2.getroot()
        objects = tree_2.findall('object')
        for object in objects:
            root_1.append(object)
            new.write(xml_1)

def main():
    os.makedirs(sys.argv[1], exist_ok=True)
    first_xmlList = os.listdir(sys.argv[2])
    for first_xml in first_xmlList:
        shutil.copy(os.path.join(sys.argv[2], first_xml), sys.argv[1])
    print("开始合并xml文件...")
    for xmlPath in sys.argv[3:]:
        for xml in os.listdir(xmlPath):
            if xml in os.listdir(sys.argv[1]):
                megerXml(os.path.join(sys.argv[1], xml), os.path.join(xmlPath, xml))
            else:
                print(xml)
                shutil.copy(os.path.join(xmlPath, xml), sys.argv[1])
    print("合并完成！")

if __name__ == '__main__':
    main()
    # lists = sys.argv[1:]
    # print(sys.argv[1], sys.argv[2])
    # print(lists)
    # shutil.copy(sys.argv[1] + "\", "new")

