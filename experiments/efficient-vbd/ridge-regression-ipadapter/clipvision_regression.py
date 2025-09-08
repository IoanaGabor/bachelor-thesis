import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
from tqdm import tqdm
import torch
import wandb
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

wandb.init(project="brain_decoding", name=f"ridge_regression_run_subj{sub}")

train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
test_fmri = np.load(test_path)

train_fmri = train_fmri/300
test_fmri = test_fmri/300


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


train_clip = torch.load('data/extracted_features_IPAdapterPlus/subj{:02d}/image_features_train.npz'.format(sub))
test_clip = torch.load('data/extracted_features_IPAdapterPlus/subj{:02d}/image_features_test.npz'.format(sub))

train_latents = [torch.stack([sublist[i][0][j] for i in range(2) for j in range(16)]) for sublist in train_clip]
test_latents = [torch.stack([sublist[i][0][j] for i in range(2) for j in range(16)]) for sublist in test_clip]

train_clip = torch.stack(train_latents).cpu().numpy().astype(np.float64)
test_clip = torch.stack(test_latents).cpu().numpy().astype(np.float64)

stds=np.zeros((32, 768)).astype(np.float64)
means=np.zeros((32, 768)).astype(np.float64)
stds_test=np.zeros((32, 768)).astype(np.float64)
means_test=np.zeros((32, 768)).astype(np.float64)
for i in tqdm(range(32)):
    means[i]=np.mean(train_clip[:,i],axis=0)
    stds[i]=np.std(train_clip[:,i],axis=0)


np.save('data/regression_weights/subj{:02d}/clipvision_statistics_train_means'.format(sub),means)
np.save('data/regression_weights/subj{:02d}/clipvision_statistics_train_stds'.format(sub),stds)

print("stats saved")


num_samples,num_embed,num_dim = len(train_clip), 32, 768

print("Training Regression")
reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float64)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float64)
pred_clip = np.zeros_like(test_clip)
train_scores=[]
test_scores=[]
for i in tqdm(range(num_embed)):
    reg = skl.Ridge(alpha=60000, max_iter=50000, fit_intercept=True)
    reg.fit(train_fmri, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(test_fmri)
    means_test[i] = np.mean(pred_test_latent, axis=0)
    stds_test[i] = np.std(pred_test_latent, axis=0)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / (1e-8+np.std(pred_test_latent,axis=0))
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)

    train_r2 = reg.score(train_fmri, train_clip[:, i])
    test_r2 = reg.score(test_fmri, test_clip[:, i])

    train_scores.append(train_r2)
    test_scores.append(test_r2)


np.save('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral.npy'.format(sub),pred_clip)

test_size = len(test_clip)

np.save('data/regression_weights/subj{:02d}/clipvision_statistics_test_means'.format(sub),means_test)
np.save('data/regression_weights/subj{:02d}/clipvision_statistics_test_stds'.format(sub),stds_test)

np.save('data/predicted_features/subj{:02d}/nsd_clipvision_regression_weights.npy'.format(sub),reg_w)

np.save('data/predicted_features/subj{:02d}/nsd_clipvision_regression_bias'.format(sub),reg_b)



with open('data/regression_weights/subj{:02d}/clipvision_regression_weights.pkl'.format(sub),"wb") as f:
  pickle.dump(datadict,f)
