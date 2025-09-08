BASE_MODEL_PATH = "SG161222/Realistic_Vision_V4.0_noVAE"
VAE_MODEL_PATH = "stabilityai/sd-vae-ft-mse"
UNCONDITIONAL_EMBEDDINGS_PATH = "ioanagabor/unconditional_embeddings.pt"

IMAGE_ENCODER_PATH = "models/image_encoder"
IP_CKPT_PATH = "models/ip-adapter-plus_sd15.bin"

RIDGE_ALPHA = 20000
RIDGE_MAX_ITER = 50000
RIDGE_FIT_INTERCEPT = True

RIDGE_WEIGHTS_PATH_TEMPLATE = 'regressors/subj{:02d}/nsd_clipvision_regression_weights.npy'
RIDGE_BIAS_PATH_TEMPLATE = 'regressors/subj{:02d}/nsd_clipvision_regression_bias.npy'

RIDGE_MEANS_TRAIN_PATH_TEMPLATE = 'statistics/subj{:02d}/clipvision_statistics_train_means.npy'
RIDGE_STDS_TRAIN_PATH_TEMPLATE = 'statistics/subj{:02d}/clipvision_statistics_train_stds.npy'
RIDGE_MEANS_TEST_PATH_TEMPLATE = 'statistics/subj{:02d}/clipvision_statistics_test_means.npy'
RIDGE_STDS_TEST_PATH_TEMPLATE = 'statistics/subj{:02d}/clipvision_statistics_test_stds.npy'
RIDGE_NORM_MEAN_TRAIN_TEMPLATE = 'statistics/subj{:02d}/norm_mean_train.npy'
RIDGE_NORM_SCALE_TRAIN_TEMPLATE = 'statistics/subj{:02d}/norm_scale_train.npy'


RIDGE_VOXEL_SCALE_FACTOR = 300  

DDIM_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "clip_sample": False,
    "set_alpha_to_one": False,
    "steps_offset": 1,
}

PIPELINE_CONFIG = {
    "torch_dtype": "float16",
    "feature_extractor": None,
    "safety_checker": None,
}

INFERENCE_CONFIG = {
    "num_inference_steps": 100,
    "num_embeddings": 16,
} 