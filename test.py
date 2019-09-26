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
    out_rgb = out_rgb.cpu().detach().numpy().reshape([3, 224, 224])
    out_rgb = out_rgb.transpose(1, 2, 0)
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb
    
def getImage(root):
    raw = Image.open(root)
    raw = np.array(raw).astype(np.float64)
    raw = (raw - 128) / 128
    raw = torch.from_numpy(raw.transpose(2, 0, 1).astype(np.float32)).contiguous()
    raw = raw.reshape([1, raw.shape[0], raw.shape[1], raw.shape[2]]).cuda()
    return raw

def generation(baseroot, saveroot, imglist, colornet):
    testtime = 0
    for i in range(len(imglist)):		#len(imglist)
		# Read raw image
        readname = baseroot + '/' + imglist[i]
        print(readname)
        # Forward propagation
        torchimg = getImage(readname)
        time1 = time.time()
        out_rgb = test(torchimg, colornet)
        time2 = time.time()
        ad_time = time2 - time1
        testtime = testtime + ad_time
        # Save
        img_rgb = Image.fromarray(out_rgb)
        savename = saveroot + '/' + imglist[i][:-4] + '.png'
        img_rgb.save(savename)
    print('total seconds:')
    print(testtime)
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
    parser.add_argument('--baseroot', type = str, default = './TestingPhoneRaw', help = 'testing images baseroot')
    parser.add_argument('--saveroot', type = str, default = './TestingResults', help = 'result images saveroot')
    opt = parser.parse_args()

    # Define the basic variables
    colornet = torch.load(opt.netroot).cuda()
    imglist = get_jpgs(opt.baseroot)
    print(imglist)

    # Generate testing set
    generation(opt.baseroot, opt.saveroot, imglist, colornet)
    
