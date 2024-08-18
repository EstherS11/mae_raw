import sys
import os
import numpy as np
from util.utils import Progbar, read_annotations
import torch.backends.cudnn as cudnn
import models_vit
import torch.utils.data
from util.tools import inference_single
import cv2
import argparse
from util.pos_embed import interpolate_pos_embed
from albumentations.pytorch.functional import mask_to_tensor


def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--model_file", type=str, help="File path to the pretrained model",default="/root/autodl-tmp/mae/output_addnoise0716")
    parser.add_argument("--test_dir", type=str, default='/root/datasets/Casiav1', help="Path to the image list")
    parser.add_argument("--test_file", type=str, default='test.txt', help="Path to the image list")
    parser.add_argument('--new_size', default=336, type=int)
    parser.add_argument("--resize", type=int, default=336)
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_opt()
    print("in the head of inference:", opt)
    cudnn.benchmark = True

    # read test data
    test_file = os.path.join(opt.test_dir, opt.test_file)
    dataset_name = os.path.basename(opt.test_dir) # CASIAv2
    # model_type = os.path.basename(opt.test_file).split('/')[-2]  # CoCoGlide
    if not os.path.exists(test_file):
        print("%s not exists,quit" % test_file)
        sys.exit()
    test_data = read_annotations(test_file)
    new_size = opt.resize

    # load model
    model_path = os.path.join(opt.model_file, 'checkpoint-best.pth')
    if os.path.exists(model_path):
        model = model = models_vit.__dict__['vit_base_patch16'](
        num_classes=1,
        drop_path_rate=0.1,
        global_pool=False,
        img_size=opt.new_size,
    )
    else:
        print("model not found ", model_path)
        sys.exit()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        print("Load well-trained checkpoint from: %s" % model_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        model.load_state_dict(checkpoint_model, strict=True)
        model.eval()
        print("load %s finish" % (os.path.basename(model_path)))
    else:
        print("%s not exist" % model_path)
        sys.exit()
    model.cuda()

    model.eval()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    save_path = os.path.join(opt.model_file,'visual_result',dataset_name)
    print("predicted maps will be saved in :%s" % save_path)
    os.makedirs(save_path, exist_ok=True)


    with torch.no_grad():
        progbar = Progbar(len(test_data), stateful_metrics=['path'])
        pd_img_lab = []
        lab_all = []
        scores = []

        for ix, (img_path, mask_path, _) in enumerate(test_data):
            img_path = os.path.join(opt.test_dir, img_path)
            img = cv2.imread(img_path)
            gt_path = os.path.join(opt.test_dir, mask_path)
            print('mask_path:', gt_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            ori_size = img.shape
            img = cv2.resize(img, (new_size, new_size))
            gt = cv2.resize(gt, (new_size, new_size))
            gt = mask_to_tensor(gt, num_classes=1, sigmoid= True).unsqueeze(0)
            gt = gt.cuda()
            seg, loss = inference_single(img=img, gt=gt, model=model, th=0)
            print('loss:', loss)


            # 定位结果图
            save_seg_path = os.path.join(save_path, os.path.split(img_path)[-1].split('.')[0] + '.png')
            os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
            seg = cv2.resize(seg, (ori_size[1], ori_size[0]))
            cv2.imwrite(save_seg_path, seg.astype(np.uint8))
            progbar.add(1, values=[('path', save_seg_path), ])


