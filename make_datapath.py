from lib import *

def make_datapath_list(rootpath):
    image_path_template = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
    annotation_path_template = os.path.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = os.path.join(rootpath, 'ImageSets/Main/train.txt')
    val_id_names = os.path.join(rootpath, 'ImageSets/Main/val.txt')

    train_img_list = list()
    train_annotation_list = list()

    for line in open(train_id_names):
        file_id = line.strip() #xóa ký tự xuống dòng, xóa space
        img_path = (image_path_template % file_id) #đưa từng file_id vào %s trên template
        anno_path = (annotation_path_template % file_id) #đưa từng file_id vào %s trên template

        train_img_list.append(img_path)
        train_annotation_list.append(anno_path)
    
    val_img_list = list()
    val_annotation_list = list()

    for line in open(val_id_names):
        file_id = line.strip() #xóa ký tự xuống dòng, xóa space
        img_path = (image_path_template % file_id) #đưa từng file_id vào %s trên template
        anno_path = (annotation_path_template % file_id) #đưa từng file_id vào %s trên template

        val_img_list.append(img_path)
        val_annotation_list.append(anno_path)
    
    return train_img_list, train_annotation_list, val_img_list, val_annotation_list

if __name__ == '__main__':
    rootpath = './data/VOCdevkit/VOC2012/'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(rootpath)
    
    print(len(train_img_list))
    print(train_img_list[0])
