'''
把coco数据集合的所有标注转换到voc格式，不改变图片命名方式，
注意，原来有一些图片是黑白照片，检测出不是 RGB 图像，这样的图像不会被放到新的文件夹中
'''
from pycocotools.coco import COCO
import os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image

# 生成图片保存的路径
# CKimg_dir = './coco2017_voc/images'
# 生成标注文件保存的路径
# CKanno_dir = './coco2017_voc/annotations'



# 若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def save_annotations(image_dir, anno_dir, filename, objs, filepath):
    CKanno_dir = anno_dir
    CKimg_dir = image_dir

    annopath = CKanno_dir + "/" + filename[:-3] + "xml"  # 生成的xml文件保存路径
    dst_path = CKimg_dir + "/" + filename
    img_path = filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        # print(filename + " not a RGB image")
        im.close()
        return 0
    im.close()
    # shutil.copy(img_path, dst_path)  # 把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    obj_tag = 0
    for obj in objs:
        if (obj[2]==obj[4] or obj[3]==obj[5]):
            print('\n error:', filename)
            obj_tag+=1

        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
        # obj_num+=1
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)

    if obj_tag > 0:
        return -1
    return filename[:-4]


def showbycv(coco, dataType, img, classes, origin_image_dir, image_dir, anno_dir, verbose=False):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir, dataType, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    result = save_annotations(image_dir, anno_dir, filename, objs, filepath)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)

    return result


def catid2name(coco):  # 将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes


def get_CK5(origin_anno_dir, origin_image_dir, image_dir, anno_dir, imgset_dir, verbose=False):
    # dataTypes = ['val2017']
    dataTypes = ['train2017_60']
    # dataTypes = ['val2017_60']
    # dataTypes = ['train2017_60', 'val2017_60']  #instances_train2017_60.json //remove categories from voc （60=80-20）
    for dataType in dataTypes:
        annFile = 'instances_{}.json'.format(dataType)
        annpath = os.path.join(origin_anno_dir, annFile)
        coco = COCO(annpath)
        classes = catid2name(coco)
        imgIds = coco.getImgIds()
        # imgIds=imgIds[0:10]#测试用，抽取10张图片，看下存储效果
        # print(imgIds[0])
        # imgIds = [
        # '000000143901', '000000324958', '000000089141', '000000274134', '000000479886', '000000024238', 
        # '000000375461', '000000544117', '000000064501', '000000450700', '000000557901', '000000111930', 
        # '000000145073', '000000315593', '000000024019', '000000145952', '000000148502', '000000378203', 
        # '000000402598', '000000561715', '000000155845', '000000462565', '000000335565', '000000012109', 
        # '000000266107', '000000372113', '000000112590', '000000332824', '000000414852', '000000555894', 
        # '000000353300', '000000008086']
        # 000000111930 有问题

        grouptxt = os.path.join(imgset_dir, '{}.txt'.format(dataType)) #保存数据分组txt
        file = open(grouptxt, 'w')

        for imgId in tqdm(imgIds):
            # imgId = imgId[6:12]
            img = coco.loadImgs(imgId)[0]
            result = showbycv(coco, dataType, img, classes, origin_image_dir, image_dir, anno_dir, verbose=verbose)
            # print(result)
            if result!=0:
                # file.write(result+'\n')
                if result == -1:
                    file.write(str(imgId) + '\n')

        file.close()


def main():
    root_dir = 'H:'
    # root_dir = '/home'

    # base_dir = './coco2017_voc'  # step1 这里是一个新的文件夹，存放转换后的图片和标注
    base_dir = root_dir + '/Datasets/VOCdevkit/VOC2017-task1-source'
    image_dir = os.path.join(base_dir, 'JPEGImages')  # 在上述文件夹中生成images，annotations两个子文件夹
    anno_dir = os.path.join(base_dir, 'Annotations')
    imgset_dir_1 = os.path.join(base_dir, 'ImageSets')
    imgset_dir = os.path.join(imgset_dir_1, 'Main')

    mkr(image_dir)
    mkr(anno_dir)
    mkr(imgset_dir_1)
    mkr(imgset_dir)
    # origin_image_dir = './coco2017'  # step 2原始的coco的图像存放位置
    # origin_anno_dir = './coco2017/annotations'  # step 3 原始的coco的标注存放位置
    origin_image_dir = root_dir + '/Datasets/COCO/022719'  # step 2原始的coco的图像存放位置
    origin_anno_dir = root_dir + '/Datasets/COCO/022719/annotations'  # step 3 原始的coco的标注存放位置
    print(origin_anno_dir)
    verbose = False  # 是否需要看下标记是否正确的开关标记，若是true,就会把标记展示到图片上
    get_CK5(origin_anno_dir, origin_image_dir, image_dir, anno_dir, imgset_dir, verbose)


if __name__ == "__main__":
    main()
