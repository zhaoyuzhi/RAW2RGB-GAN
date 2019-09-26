import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import color

import network

# ----------------------------------------
#                H, W Data
# ----------------------------------------
#  item      H      W
#  1.png    1509   2052
#  2.png    1508   2067
#  3.png    1508   2065
#  4.png    1509   2050
#  5.png    1509   2052
#  6.png    1487   2082
#  7.png    1496   2046
#  8.png    1506   2039
#  9.png    1509   2081
#  10.png   1503   2034
#   MIN     1344   2016

# ----------------------------------------
#                 Testing
# ----------------------------------------

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

def test(rgb, colornet):
    out_rgb, fake_sal = colornet(rgb)
    out_rgb = out_rgb.cpu().detach().numpy().reshape([3, 768, 768])
    out_rgb = out_rgb.transpose(1, 2, 0)
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb
    
def getImage(root):
    raw = Image.open(root)
    #templist = []
    #raw = raw.crop((224, 224, 448, 448))
    raw = np.array(raw).astype(np.float64)
    raw = (raw - 128) / 128
    raw = torch.from_numpy(raw.transpose(2, 0, 1).astype(np.float32)).contiguous()
    raw = raw.reshape([1, raw.shape[0], raw.shape[1], raw.shape[2]]).cuda()
    return raw

def visualize(templist, height, width):
    img = np.zeros((height, width, 3), dtype = np.uint8)
    print(len(templist))
    for w in range(2):
        for h in range(1):
            img[(h * 768):(h * 768 + 768), (w * 768):(w * 768 + 768), :] = templist[w + h]
    for w in range(2):
        img[(height - 768):height, (w * 768):(w * 768 + 768), :] = templist[2 + w]
    for h in range(6):
        img[(h * 768):(h * 768 + 768), (width - 768):width, :] = templist[2 + 2 + h]
    img[(height - 768):height, (width - 768):width, :] = templist[-1]
    return img

def generation(baseroot, saveroot, imglist, colornet):
    for i in range(len(imglist)):		#len(imglist)
        templist = []
		# Read raw image
        readname = baseroot + '/' + imglist[i]
        print(readname)
        # Forward propagation
        torchimg = getImage(readname)
        height = torchimg.size(2)
        width = torchimg.size(3)
        # Crop to many 224 * 224
        for w in range(2):
            for h in range(1):
                inimg = torchimg[:, :, (h * 768):(h * 768 + 768), (w * 768):(w * 768 + 768)]
                out_rgb = test(inimg, colornet)
                templist.append(out_rgb)
        for w in range(2):
            inimg = torchimg[:, :, (height - 768):height, (w * 768):(w * 768 + 768)]
            out_rgb = test(inimg, colornet)
            templist.append(out_rgb)
        for h in range(1):
            inimg = torchimg[:, :, (h * 768):(h * 768 + 768), (width - 768):width]
            out_rgb = test(inimg, colornet)
            templist.append(out_rgb)
        inimg = torchimg[:, :, (height - 768):height, (width - 768):width]
        out_rgb = test(inimg, colornet)
        templist.append(out_rgb)
        # Visualize
        img = visualize(templist, height, width)
        # Save
        img_rgb = Image.fromarray(img)
        savename = saveroot + '/' + imglist[i][:-4] + '.png'
        img_rgb.save(savename)
    print('Done!')

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--netroot', type = str, default = 'zyz987.pth', help = 'model root')
    parser.add_argument('--baseroot', type = str, default = './FullResTestingPhoneRaw', help = 'validation images baseroot')
    parser.add_argument('--saveroot', type = str, default = './FullResResults', help = 'result images saveroot')
    opt = parser.parse_args()

    # Define the basic variables
    colornet = torch.load(opt.netroot).cuda()
    imglist = get_jpgs(opt.baseroot)
    print(imglist)

    generation(opt.baseroot, opt.saveroot, imglist, colornet)
    