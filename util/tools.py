import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from util.transforms import direct_val
import pdb
debug = 0


transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


def run_model(model, inputs, gt):
    output, loss = model(inputs, gt)
    return output, loss


def inference_single(img, gt, model, th=0):
    model.eval()
    with torch.no_grad():
        img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
        img = direct_val(img)
        img = img.cuda()
        seg, loss = run_model(model, img, gt)
        seg = seg.detach().cpu()
        if torch.isnan(seg).any() or torch.isinf(seg).any():
            max_score = 0.0
        else:
            max_score = torch.max(seg).numpy()
        seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]


        if len(seg) != 1:
            pdb.set_trace()
        else:
            fake_seg = seg[0]
        if th == 0:
            # return fake_seg, max_score, pb_map[0], th_map[0]
            return fake_seg, loss
        fake_seg = 255.0 * (fake_seg > 255 * th)
        fake_seg = fake_seg.astype(np.uint8)
        # pb_map = 255.0 * (pb_map > 255 * th)
        # pb_map = pb_map.astype(np.uint8)
        # th_map = 255.0 * (th_map > 255 * th)
        # th_map = th_map.astype(np.uint8)


    return fake_seg, loss

