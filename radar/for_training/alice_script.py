'''
辅助脚本 ： 从官方数据集下只保留car标签
作者：truth
最后修改:2020.0722
'''
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET

xmldir = '/home/truth/ClionProjects/mySITP/DJI/DJI ROCO/my_data_set2/annotation'

for xmlfile in os.listdir(xmldir):
    xmlname = os.path.splitext(xmlfile)[0]

    if len(xmlname) < 20:
        continue
    #从xml中读取，使用getroot()获取根节点 得到一个element对象
    tree = ET.parse(xmlname+'.xml')
    root = tree.getroot()

    #tag = element.text #访问Element标签
    #attrib = element.attrib #访问Element属性
    #text = element.text #访问Element文本

    for element in root.findall('object'):
        tag = element.tag #访问Element标签
        attrib = element.attrib #访问Element属性
        text = element.find('name').text #访问Element文本

        if text != 'car':
            root.remove(element)
        else:
            print(tag, attrib, text)

    tree.write(xmlname+'.xml')
    #print(root[0][0].text) #子节点是嵌套的，我们可以通关索引访问特定的子节点
