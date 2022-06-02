import argparse
from utils.util import *
from trainers.eval import meta_test


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_way", type=int, default=5)
    parser.add_argument("--train_shot", type=int, default=5)
    parser.add_argument("--train_query_shot", type=int, default=15)
    parser.add_argument("--gpu", help="gpu device", type=int, default=0)
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--model", choices=['Proto', 'FRN'])
    parser.add_argument("--dataset", choices=['cub_cropped', 'cub_raw',
                                              'aircraft',
                                              'meta_iNat', 'tiered_meta_iNat'])

    parser.add_argument("--TDM", action="store_true")
    parser.add_argument("--drop", action="store_true")
    parser.add_argument("--warm_up", action="store_true")

    args = parser.parse_args()

    return args


args = test_parser()
if args.dataset =='tiered_ImageNet':
    test_path = dataset_path(args)
    test_path = os.path.join(test_path, 'test')
else:
    test_path = dataset_path(args)
    test_path = os.path.join(test_path, 'test_pre')
save_path = get_save_path(args)
args.save_path = save_path
if args.dataset == 'mini_ImageNet':
    args.save_path = '/project/ssd0/subeen/CVPR2022_CTX2/Fine_grained/FRN/mini_ImageNet/TDM_finetune_drop/ResNet-12_5-shot'
if args.dataset == 'tiered_ImageNet':
    args.save_path = '/project/ssd0/subeen/CVPR2022_CTX2/Fine_grained/FRN/tiered_ImageNet/TDM_finetune_drop/ResNet-12_5-shot'
logger_path = os.path.join(args.save_path, 'test.log')
if os.path.isfile(logger_path):
    file = open(logger_path, 'r')
    lines = file.read().splitlines()
    file.close()
    logger = get_logger(logger_path)
    for i in range(len(lines)):
        logger.info(lines[i][17:])
else:
    logger = get_logger(logger_path)

model = load_pretrained_model(args)
torch.cuda.set_device(args.gpu)
model.cuda()
model.eval()

with torch.no_grad():
    way = 5
    for shot in [1, 5]:
        if args.dataset =='tiered_ImageNet':
            pre = False
            transform_type = 2
        else:
            pre = True
            transform_type = None

        mean, interval = meta_test(data_path=test_path,
                                   model=model,
                                   way=way,
                                   shot=shot,
                                   pre=pre,
                                   transform_type=transform_type,
                                   trial=1000,
                                   warm_up=False,
                                   multi_gpu=False)
        logger.info('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))