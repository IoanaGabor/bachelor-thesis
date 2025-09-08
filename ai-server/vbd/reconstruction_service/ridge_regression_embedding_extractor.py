import numpy as np
from reconstruction_service.abstract_voxel_embedding_extractor import AbstractVoxelEmbeddingExtractor
from sklearn.linear_model import Ridge
from reconstruction_service.constants import (
    RIDGE_ALPHA,
    RIDGE_MAX_ITER,
    RIDGE_FIT_INTERCEPT,
    RIDGE_WEIGHTS_PATH_TEMPLATE,
    RIDGE_BIAS_PATH_TEMPLATE,
    RIDGE_MEANS_TRAIN_PATH_TEMPLATE,
    RIDGE_STDS_TRAIN_PATH_TEMPLATE,
    RIDGE_MEANS_TEST_PATH_TEMPLATE,
    RIDGE_STDS_TEST_PATH_TEMPLATE,
    RIDGE_NORM_MEAN_TRAIN_TEMPLATE,
    RIDGE_NORM_SCALE_TRAIN_TEMPLATE,
    RIDGE_VOXEL_SCALE_FACTOR
)
from reconstruction_service.logger_config import logger

class RidgeRegressionVoxelEmbeddingExtractor(AbstractVoxelEmbeddingExtractor):
    
    def extract(self, voxel_data, person_id: str) -> np.ndarray:
        weights_path = RIDGE_WEIGHTS_PATH_TEMPLATE.format(person_id)
        bias_path = RIDGE_BIAS_PATH_TEMPLATE.format(person_id)
        
        means_train_path = RIDGE_MEANS_TRAIN_PATH_TEMPLATE.format(person_id)
        stds_train_path = RIDGE_STDS_TRAIN_PATH_TEMPLATE.format(person_id)
        means_test_path = RIDGE_MEANS_TEST_PATH_TEMPLATE.format(person_id)
        stds_test_path = RIDGE_STDS_TEST_PATH_TEMPLATE.format(person_id)
        norm_mean_train = RIDGE_NORM_MEAN_TRAIN_TEMPLATE.format(person_id)
        norm_scale_train = RIDGE_NORM_SCALE_TRAIN_TEMPLATE.format(person_id)
        
        mean_train = np.load(norm_mean_train)
        scale_train = np.load(norm_scale_train)
        
        voxel_data = np.array(voxel_data, dtype=np.float64)
        voxel_data = voxel_data / RIDGE_VOXEL_SCALE_FACTOR
        voxel_data = (voxel_data - mean_train) / scale_train
        
        reg_w = np.load(weights_path)
        reg_b = np.load(bias_path)
        means_train = np.load(means_train_path)
        stds_train = np.load(stds_train_path)
        means_test = np.load(means_test_path)
        stds_test = np.load(stds_test_path)
        
        num_embed, num_dim, _ = reg_w.shape
        embeddings = np.zeros((num_embed, num_dim)).astype(np.float64)
        
        for i in range(num_embed):
            ridge = Ridge(
                alpha=RIDGE_ALPHA,
                max_iter=RIDGE_MAX_ITER,
                fit_intercept=RIDGE_FIT_INTERCEPT
            )
            ridge.coef_ = reg_w[i]
            ridge.intercept_ = reg_b[i]
            pred = ridge.predict(voxel_data.reshape(1, -1))
            
            if stds_test[i].all() != 0:
                std_norm_test_latent = (pred - means_test[i]) / stds_test[i]
                embeddings[i] = std_norm_test_latent * stds_train[i] + means_train[i]
            else:
                embeddings[i] = pred

        return embeddings
