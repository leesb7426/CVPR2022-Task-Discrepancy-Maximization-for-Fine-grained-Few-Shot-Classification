from PIL import Image
from tqdm import tqdm
import torch
import yaml
import os
import logging
import torchvision.transforms as transforms

from models.Proto import Proto
from models.FRN import FRN

def mkdir(path):
    
    if os.path.exists(path): 
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


# get pre-resized 84x84 images for validation and test
def get_pre_folder(image_folder,transform_type):
    split = ['val','test']

    if transform_type == 0:
        transform = transforms.Compose([transforms.Resize(92),
                                    transforms.CenterCrop(84)])
    elif transform_type == 1:
        transform = transforms.Compose([transforms.Resize([92,92]),
                                    transforms.CenterCrop(84)])

    cat_list = []

    for i in split:
        
        cls_list = os.listdir(os.path.join(image_folder,i))

        folder_name = i+'_pre'

        mkdir(os.path.join(image_folder,folder_name))

        for j in tqdm(cls_list):

            mkdir(os.path.join(image_folder,folder_name,j))

            img_list = os.listdir(os.path.join(image_folder,i,j))

            for img_name in img_list:
        
                img = Image.open(os.path.join(image_folder,i,j,img_name))
                img = img.convert('RGB')
                img = transform(img)
                img.save(os.path.join(image_folder,folder_name,j,img_name[:-3]+'png'))


def get_device_map(gpu):
    cuda = lambda x: 'cuda:%d'%x
    temp = {}
    for i in range(4):
        temp[cuda(i)]=cuda(gpu)
    return temp


def dataset_path(args):
    with open('config.yml', 'r') as f:
        temp = yaml.safe_load(f)
    data_path = os.path.abspath(temp['data_path'])

    if args.dataset == 'cub_cropped':
        fewshot_path = os.path.join(data_path, 'CUB_fewshot_cropped')
    elif args.dataset == 'cub_raw':
        fewshot_path = os.path.join(data_path, 'CUB_fewshot_raw')
    elif args.dataset == 'aircraft':
        fewshot_path = os.path.join(data_path, 'Aircraft_fewshot')
    elif args.dataset == 'meta_iNat':
        fewshot_path = os.path.join(data_path, 'meta_iNat')
    elif args.dataset == 'tiered_meta_iNat':
        fewshot_path = os.path.join(data_path, 'tiered_meta_iNat')
    elif args.dataset == 'stanford_car':
        fewshot_path = os.path.join(data_path, 'StanfordCar')
    elif args.dataset == 'stanford_dog':
        fewshot_path = os.path.join(data_path, 'StanfordDog')

    return fewshot_path


def load_model(args):
    if args.model == 'FRN':
        model = FRN(args=args)
    elif args.model == 'Proto':
        model = Proto(args=args)

    return model


def get_save_path(args):
    path = os.getcwd()
    path = os.path.join(path, 'Fine_grained')
    path = os.path.join(path, args.model)

    if args.dataset == 'cub_cropped':
        path = os.path.join(path, 'CUB_fewshot_cropped')
    elif args.dataset == 'cub_raw':
        path = os.path.join(path, 'CUB_fewshot_raw')
    elif args.dataset == 'aircraft':
        path = os.path.join(path, 'Aircraft_fewshot')
    elif args.dataset == 'meta_iNat':
        path = os.path.join(path, 'meta_iNat')
    elif args.dataset == 'tiered_meta_iNat':
        path = os.path.join(path, 'tiered_meta_iNat')
    elif args.dataset == 'stanford_car':
        path = os.path.join(path, 'StanfordCar')
    elif args.dataset == 'stanford_dog':
        path = os.path.join(path, 'StanfordDog')

    if args.TDM:
        detail_path = 'TDM'
    else:
        detail_path = 'OG'

    detail_path = detail_path.replace("/", "_")
    path = os.path.join(path, detail_path)

    if args.resnet:
        backbone_path = 'ResNet-12' + '_' + str(args.train_shot) + '-shot'
    else:
        backbone_path = 'Conv-4' + '_' + str(args.train_shot) + '-shot'
    path = os.path.join(path, backbone_path)

    return path


def load_resume_point(args, model):
    if args.resnet:
        name = 'ResNet-12'
        load_path = os.path.join(args.save_folder, 'model_%s.pth' % (name))
    else:
        name = 'Conv-4'
        load_path = os.path.join(args.save_folder, 'model_%s.pth' % (name))

    try:
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
    except:
        loaded_model = torch.jit.load(load_path, map_location='cpu')
        model.load_state_dict(loaded_model.state_dict())
    return model


def load_pretrained_model(args):
    if args.model == 'FRN':
        model = FRN(args=args)
    elif args.model == 'Proto':
        model = Proto(args=args)

    if args.resnet:
        load_path = os.path.join(args.save_path, 'model_ResNet-12.pth')
    else:
        load_path = os.path.join(args.save_path, 'model_Conv-4.pth')

    try:
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
    except:
        loaded_model = torch.jit.load(load_path, map_location='cpu')
        model.load_state_dict(loaded_model.state_dict())
    return model


def get_logger(filename):

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
