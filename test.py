import argparse
from utils.util import *
from trainers.eval import meta_test


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", nargs='+', help="number of support images per class for meta-testing during final test", type=int)
    parser.add_argument("--query", type=int, default=16)
    parser.add_argument("--gpu_num", help="gpu device", type=int, default=1)
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--model", choices=['Proto', 'FRN'])
    parser.add_argument("--dataset", choices=['cub_cropped', 'cub_raw',
                                              'aircraft',
                                              'meta_iNat', 'tiered_meta_iNat',
                                              'stanford_car', 'stanford_dog'])
    parser.add_argument("--TDM", action="store_true")

    return args


args = test_parser()
assert args.gpu_num > 0, "TDM is only tested with GPU setting"

test_path = dataset_path(args)
test_path = os.path.join(test_path, 'test_pre')

save_path = get_save_path(args)
args.save_path = save_path
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
model.eval()
model.cuda()
if args.gpu_num > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(args.gpu_num)))

with torch.no_grad():
    for shot in args.shot:
        pre = True
        transform_type = None

        mean, interval = meta_test(data_path=test_path,
                                   model=model,
                                   way=args.way,
                                   shot=shot,
                                   pre=pre,
                                   transform_type=transform_type,
                                   multi_gpu=args.gpu_num,
                                   query_shot=args.query,
                                   trial=1000,)
        logger.info('%d-way-%d-shot acc: %.3f\t%.3f' % (args.way, shot, mean, interval))