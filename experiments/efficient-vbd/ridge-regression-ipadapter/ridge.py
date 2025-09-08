import sys
import numpy as np
import sklearn.linear_model as skl
import utils.data as data
import pickle
import argparse
from tqdm import tqdm
import torch
import wandb
import os
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
data_path="./data/natural-scenes-dataset"
subject=1
model_path = "./logs/model/client{subject}_best.pth".format(subject=subject)
batch_size=1
num_workers=1
seed=42
pool_type='max'
pool_num=8192
length=982
train_path = "{}/webdataset_avg_split/train/train_subj0{}".format(data_path, subject)
val_path = "{}/webdataset_avg_split/val/val_subj0{}".format(data_path, subject)
test_path = "{}/webdataset_avg_split/test/test_subj0{}".format(data_path, subject)
description = ""
original_output_dir="./outputs_original/test/test_subj0{}".format(subject)
recreated_output_dir="./outputs_recreated/{}/test/test_subj0{}".format(description, subject)
extensions_train = ['nsdgeneral.npy', "jpg", 'coco73k.npy', 'ip']
extensions_test = ['nsdgeneral.npy', "jpg", 'coco73k.npy', 'ip']
original_output_dir="./outputs_original_ridge/test/test_subj0{}".format(subject)
os.makedirs(os.path.dirname(original_output_dir), exist_ok=True)
train_dl = data.get_dataloader(
    train_path,
    batch_size=1,
    num_workers=num_workers,
    seed=seed,
    extensions=extensions_train,
    pool_num=None,
    is_shuffle=True,
)

test_dl = data.get_dataloader(
    train_path,
    batch_size=1,
    num_workers=num_workers,
    seed=seed,
    extensions=extensions_test,
    pool_num=None,
    is_shuffle=True,
)

voxel_list = []
ip_list = []
cnt = 0
for train_i, data_i in tqdm(enumerate(train_dl)):
    voxel, image, coco, ip = data_i
    voxel = torch.mean(voxel, axis=1).float()
    voxel_list.append(voxel.cpu().numpy())
    ip_list.append(ip.cpu().numpy())
train_fmri = np.concatenate(voxel_list, axis=0)
train_clip = np.concatenate(ip_list, axis=0)
train_fmri = train_fmri / 300

voxel_list = []
cnt = 0
ip_list = []
idx = 0
for test_i, data_i in tqdm(enumerate(test_dl)):
    voxel, image, coco, ip = data_i
    voxel = torch.mean(voxel, axis=1).float()
    voxel_list.append(voxel.cpu().numpy())
    ip_list.append(ip.cpu().numpy())    
    original_path = os.path.join(original_output_dir, f"{idx}.original.jpg")
    image.save(original_path)
    idx += 1
test_fmri = np.concatenate(voxel_list, axis=0)
test_fmri = test_fmri / 300
test_clip = np.concatenate(ip_list, axis=0)

norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
print(norm_mean_train)
print(norm_scale_train)

np.save('norm_mean_train'.format(sub),norm_mean_train)
np.save('norm_scale_train'.format(sub),norm_scale_train)

train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print(np.mean(train_fmri),np.std(train_fmri))
print(np.mean(test_fmri),np.std(test_fmri))

print(np.max(train_fmri),np.min(train_fmri))
print(np.max(test_fmri),np.min(test_fmri))

num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)

train_clip = torch.tensor(train_clip)
test_clip = torch.tensor(test_clip)

print(train_clip.shape)
print(test_clip.shape)
train_latents = [torch.stack([torch.tensor(sublist[0][j])  for j in range(16)]) for sublist in train_clip]
test_latents = [torch.stack([torch.tensor(sublist[0][j]) for j in range(16)]) for sublist in test_clip]

train_clip = torch.stack(train_latents).cpu().numpy().astype(np.float64)
test_clip = torch.stack(test_latents).cpu().numpy().astype(np.float64)

stds=np.zeros((16, 768)).astype(np.float64)
means=np.zeros((16, 768)).astype(np.float64)
stds_test=np.zeros((16, 768)).astype(np.float64)
means_test=np.zeros((16, 768)).astype(np.float64)
for i in tqdm(range(16)):
    means[i]=np.mean(train_clip[:,i],axis=0)
    stds[i]=np.std(train_clip[:,i],axis=0)

print("stats saved")


num_samples,num_embed,num_dim = len(train_clip), 16, 768

print("Training Regression")
reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float64)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float64)
pred_clip = np.zeros_like(test_clip)
train_scores=[]
test_scores=[]
alpha_value = 60000
for i in tqdm(range(num_embed)):
    reg = skl.Ridge(alpha=alpha_value, max_iter=50000, fit_intercept=True)
    reg.fit(train_fmri, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    pred_test_latent = reg.predict(test_fmri)
    means_test[i] = np.mean(pred_test_latent, axis=0)
    stds_test[i] = np.std(pred_test_latent, axis=0)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / (1e-8+np.std(pred_test_latent,axis=0))
    print(pred_clip.shape)
    print(std_norm_test_latent.shape)
    print(np.std(train_clip[:,i],axis=0).shape)
    print(np.mean(train_clip[:,i],axis=0).shape)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    train_r2 = reg.score(train_fmri, train_clip[:, i])
    test_r2 = reg.score(test_fmri, test_clip[:, i])
    train_scores.append(train_r2)
    test_scores.append(test_r2)

output_dir = f'regression_weights_{alpha_value}/subj{sub}'
os.makedirs(output_dir, exist_ok=True)
np.save(f'regression_weights_{alpha_value}/subj{sub}/pred_clip.npy', pred_clip)

with open(f'regression_weights_{alpha_value}/subj{sub}/clipvision_regression_weights.pkl',"wb") as f:
  pickle.dump(datadict,f)



