from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

def read_lab(path):
    img = Image.open(path)
    img = np.array(img,np.uint8)
    return img

def parse_xml(FILENAME):
    inst_info = [[0,0,0,0,0,0]]
    tree = ET.parse(FILENAME)
    root = tree.getroot()
    for child in root:
        if child.tag == 'object':
            for c in child:
                if c.tag == 'bndbox':
                    inst_buff = []
                    for c2 in c:
                        inst_buff.append(int(c2.text))
                    xcenter = (inst_buff[0]+inst_buff[2]) //2
                    ycenter = (inst_buff[1]+inst_buff[3]) //2
                    inst_buff.append(xcenter)
                    inst_buff.append(ycenter)
                    inst_info.append(inst_buff)
    return inst_info

def get_coord_map(FILENAME):
    lab_name = FILENAME.replace('JPEGImages','SegmentationObject').replace('.jpg','.png')
    annot_name = FILENAME.replace('JPEGImages','Annotations').replace('.jpg','.xml')

    img = read_lab(lab_name)
    img[img==255] = 0

    lab = parse_xml(annot_name)

    lab = np.int32(lab)
    img = np.int32(img)
    # print(lab)
    # input()
    coord = lab[img]
    # print(coord[250,240:250])
    # input()
    return coord

def get_mask(FILENAME):
    lab_name = FILENAME.replace('JPEGImages','SegmentationObject').replace('.jpg','.png')
    img = read_lab(lab_name)
    img[img==255] = 0
    img[img!=0] = 1
    return img

def get_seg(FILENAME):
    lab_name = FILENAME.replace('JPEGImages','SegmentationClass').replace('.jpg','.png')
    res = read_lab(lab_name)
    res[res==255] = 0
    return res

class_list = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
              'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
class_dict = {}
for i in range(len(class_list)):
    class_dict[class_list[i]] = i

def get_inst_num(FILENAME):
    annot_name = FILENAME.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')

    res = np.zeros([20],np.float32)
    tree = ET.parse(annot_name)
    root = tree.getroot()
    for child in root:
        if child.tag == 'object':
            for c in child:
                if c.tag == 'name':
                    name = c.text
                    res[class_dict[name]] += 1
    return res

