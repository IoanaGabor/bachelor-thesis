
# This file is an adaptation of https://github.com/littlepure2333/MindBridge/blob/main/src/eval.py. 


import os
import numpy as np
import math
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from options import args
from PIL import Image
import random
from torchvision.models.feature_extraction import create_feature_extractor

import pdb


def load_image(file_path):
    img = Image.open(file_path)
    transform_image = transforms.ToTensor()
    image_tensor = transform_image(img)
    return image_tensor


def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")


def _validate_weights(w, dtype=np.float64):
    w = _validate_vector(w, dtype=dtype)
    if np.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


def correlation(u, v, w=None, centered=True):
    u = _validate_vector(u)
    v = _validate_vector(v)
    if w is not None:
        w = _validate_weights(w)
        w = w / w.sum()
    if centered:
        if w is not None:
            umu = np.dot(u, w)
            vmu = np.dot(v, w)
        else:
            umu = np.mean(u)
            vmu = np.mean(v)
        u = u - umu
        v = v - vmu
    if w is not None:
        vw = v * w
        uw = u * w
    else:
        vw, uw = v, u
    uv = np.dot(u, vw)
    uu = np.dot(u, uw)
    vv = np.dot(v, vw)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    return np.clip(dist, 0.0, 2.0)


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print('Note: not using cudnn.deterministic')


@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True, device='cpu'):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1

def cal_metrics(original_stimuli_path, results_path, device):
    original_files = [f for f in os.listdir(original_stimuli_path)]
    results_files = [f for f in os.listdir(results_path)]

    number = 0
    all_images, all_brain_recons = None, None
    all_images, all_brain_recons = [], []
    for file in tqdm(original_files):
        image = load_image(file)
        all_images.append(image)

    for file in tqdm(results_files):
        image = load_image(file)
        all_brain_recons.append(image)

    all_images = torch.vstack(all_images)
    all_brain_recons = torch.vstack(all_brain_recons)
    all_images = all_images.to(device)
    all_brain_recons = all_brain_recons.to(device).to(all_images.dtype).clamp(0,1).squeeze()

    print("Images shape:", all_images.shape)
    print("Recons shape:", all_brain_recons.shape)
    print("Number:", number)

    ### PixCorr
    print("\n------calculating pixcorr------")
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()

    print(all_images_flattened.shape)
    print(all_brain_recons_flattened.shape)

    corrsum = 0
    for i in tqdm(range(number)):
        corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
    corrmean = corrsum / number

    pixcorr = corrmean
    print(pixcorr)

    del all_images_flattened
    del all_brain_recons_flattened
    torch.cuda.empty_cache()

    ### SSIM
    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
    print("\n------calculating ssim------")

    ssim_score=[]
    for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
        ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    ssim = np.mean(ssim_score)
    print(ssim)


    #### AlexNet
    from torchvision.models import alexnet, AlexNet_Weights
    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    layer = 'early, AlexNet(2)'
    print(f"\n------calculating {layer}------")
    all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                             alex_model, preprocess, 'features.4', device=device)
    alexnet2 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet2:.4f}")

    layer = 'mid, AlexNet(5)'
    print(f"\n------calculating {layer}------")
    all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                             alex_model, preprocess, 'features.11', device=device)
    alexnet5 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet5:.4f}")

    del alex_model
    torch.cuda.empty_cache()

    #### InceptionV3
    print(f"\n------calculating Inception------")
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            inception_model, preprocess, 'avgpool', device=device)
            
    inception = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {inception:.4f}")

    del inception_model
    torch.cuda.empty_cache()


    #### CLIP
    print(f"\n------calculating CLIP------")
    import clip
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            clip_model.encode_image, preprocess, None, device=device) # final layer
    clip_ = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {clip_:.4f}")


    del clip_model
    torch.cuda.empty_cache()


    #### Efficient Net
    print(f"\n------calculating Efficient Net------")
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool']).to(device)
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()
    effnet = np.array([correlation(gt[i], fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",effnet)


    del eff_model
    torch.cuda.empty_cache()


    #### SwAV
    print(f"\n------calculating SwAV------")
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    swav = np.array([correlation(gt[ii],fake[ii]) for ii in range(len(gt))]).mean()
    print("Distance:",swav,"\n")


    del swav_model
    torch.cuda.empty_cache()

    data = {
        "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"],
        "Value": [pixcorr, ssim, alexnet2, alexnet5, inception, clip_, effnet, swav],
    }
    print(results_path)
    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(results_path, f'_metrics_on_{number}samples.csv'), sep='\t', index=False)

if __name__ == "__main__":
    seed_everything(seed=args.seed)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    cal_metrics(args.original_stimuli_path, args.results_path, device)