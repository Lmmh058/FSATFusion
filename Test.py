from PIL import Image
import numpy as np
import os
import torch
import time
import imageio
import torchvision.transforms as transforms
from Networks.network import MODEL as net
import statistics
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')

method = 'Final'

model = net(in_channel=2)

model_path = "./models/abc/model_48.pth"

model = model.cuda()

model.load_state_dict(torch.load(model_path))


def fusion():
    fuse_time = []
    for num in range(1,1606):
        path1 = './Source_image_TNO/ir/{}.bmp'.format(num)
        path2 = './Source_image_TNO/vi/{}.bmp'.format(num)

        # 检查文件是否存在于 path1 和 path2
        path1_exists = os.path.exists(path1)
        path2_exists = os.path.exists(path2)

        if not path1_exists and not path2_exists:
            print(f'Warning: File {num} not found in both {path1} and {path2}. Skipping to next file.')
            continue  # 自动 num 加 1，继续循环
        elif not path1_exists or not path2_exists:
            missing_path = path1 if not path1_exists else path2
            raise FileNotFoundError(f'Error: {missing_path} not found. Exiting.')

        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2)
        start = time.time()
        if img2.mode == 'RGB':
            img2 = np.array(img2)
            img2_yuv = cv2.cvtColor(img2, cv2.COLOR_RGB2YUV)
            img2_y = img2_yuv[:, :, 0]
        else:
            img2_y = img2
            img2_yuv = None

        tran = transforms.ToTensor()

        img1_org = tran(img1)
        img2_org = tran(img2_y)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)

        input_img = input_img.cuda()

        model.eval()
        out = model(input_img)
        result = np.squeeze(out.detach().cpu().numpy())
        result = (result * 255).astype(np.uint8)


        if img2_yuv is not None:
            img2_yuv[:, :, 0] = result
            fused_img = cv2.cvtColor(img2_yuv, cv2.COLOR_YUV2RGB)
            imageio.imwrite('./fusion results/TNO_yanzheng/{}_FSATFusionyanzheng.bmp'.format(num), fused_img)
        else:
            imageio.imwrite('./fusion results/TNO_yanzheng/{}_FSATFusionyanzheng.bmp'.format(num), result)
        end = time.time()
        fuse_time.append(end - start)

    mean = statistics.mean(fuse_time)
    print(f'fuse avg time: {mean:.4f}')


if __name__ == '__main__':
    fusion()