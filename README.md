# Saiency Map-aided GAN for RAW2RGB Mapping

The PyTorch implementations and guideline for Saiency Map-aided GAN for RAW2RGB Mapping in `2019 IEEE ICCV Workshop in Advanced Image Manipulation (AIM)`.

## 1 Implementations

Before running it, please ensure the environment is `Python 3.6` and `PyTorch 1.0.1`.

### 1.1  Train

If you train it from scratch, please download the saliency map generated by our pre-trained [SalGAN](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_ad_cityu_edu_hk/EYH8wcYdU7xKjksv9HJIa2oBm7W0702P2_vPnDMv8Jt3Rg?e=yIEV8M).

Stage 1:
```bash
python train.py     --in_root [the path of TrainingPhoneRaw]
		    --out_root [the path of TrainingCanonRGB]
		    --sal_root [the path of TrainingCanonRGB_saliency]
```
Stage 2:
```bash
python train.py     --epochs 30
                    --lr_g 0.0001
                    --in_root [the path of TrainingPhoneRaw]
                    --out_root [the path of TrainingCanonRGB]
                    --sal_root [the path of TrainingCanonRGB_saliency]
```
```bash
if you have more than one GPU, please change following codes:
python train.py     --multi_gpu True
                    --gpu_ids [the ids of your multi-GPUs]
```

The training pairs are shown like this:

<img src="./images/train.png" width="800"/>

The model is shown as:

<img src="./images/architecture.png" width="800"/>

### 1.1  Test

At testing phase, please create a folder first if the folder is not exist.

Please download the pre-trained [model](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_ad_cityu_edu_hk/EQw2dl2a2VBMmGpgpK7oBDYBcAnVitgCrhUyORRUcqW4aQ?e=iWsk3c) first.

For small image patches:
```bash
python test.py 	    --netroot 'zyz987.pth' (please ensure the pre-trained model is in same path)
		    --baseroot [the path of TestingPhoneRaw]
		    --saveroot [the path that all the generated images will be saved to]
```

For full resolution images:
```bash
python test_full_res.py
or python test_full_res2.py
--netroot 'zyz987.pth' (please ensure the pre-trained model is in same path)
--baseroot [the path of FullResTestingPhoneRaw]
--saveroot [the path that all the generated images will be saved to]
```

Some randomly selected patches are shown as:

<img src="./images/patch.png" width="800"/>

### 2 Comparison with Pix2Pix

We have trained a Pix2Pix framework using same settings.

Because both systems are trained only with L1 loss at first stage, the generated samples are obviously more blurry than second stage. There is artifact in the images produced by Pix2Pix due to Batch Normalization. Moreover, we show the results produced by proposed architecture trained only with L1 loss for 40 epochs. Note that, our proposed system are optimized by whole objectives for last 30 epochs. It demonstrates that adversarial training and perceptual loss indeed enhance visual quality.

<img src="./images/val.png" width="800"/>

### 3 Full resolution results

Because the memory is not enough for generate a high resolution image, we alternatively generate patch-by-patch.

<img src="./images/1.png" width="800"/>

<img src="./images/2.png" width="800"/>

<img src="./images/3.png" width="800"/>

<img src="./images/4.png" width="800"/>

<img src="./images/5.png" width="800"/>

<img src="./images/6.png" width="800"/>

<img src="./images/7.png" width="800"/>

<img src="./images/8.png" width="800"/>

<img src="./images/9.png" width="800"/>

<img src="./images/10.png" width="800"/>

## 3 Acknowledgement

If you have any question, please do not hesitate to contact yzzhao2-c@my.cityu.edu.hk
