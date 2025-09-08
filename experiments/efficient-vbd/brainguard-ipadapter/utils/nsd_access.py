# This file is an adaptation of https://github.com/kunzhan/BrainGuard/blob/main/utils/nsd_access.py

import os
import os.path as op
import glob
import nibabel as nb
import numpy as np
import pandas as pd
from pandas import json_normalize
import h5py
import matplotlib.pyplot as plt


class NSDAccess(object):

    def __init__(self, nsd_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsd_folder = nsd_folder
        self.nsddata_folder = op.join(self.nsd_folder, 'nsddata')
        self.ppdata_folder = op.join(self.nsd_folder, 'nsddata', 'ppdata')
        self.nsddata_betas_folder = op.join(
            self.nsd_folder, 'nsddata_betas', 'ppdata')

        self.behavior_file = op.join(
            self.ppdata_folder, '{subject}', 'behav', 'responses.tsv')
        self.stimuli_file = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5')
        self.stimuli_description_file = op.join(
            self.nsd_folder, 'nsddata', 'experiments', 'nsd', 'nsd_stim_info_merged.csv')

    def affine_header(self, subject, data_format='func1pt8mm'):

        full_path = op.join(self.ppdata_folder,
                            '{subject}', '{data_format}', 'brainmask.nii.gz')
        full_path = full_path.format(subject=subject,
                                     data_format=data_format)
        nii = nb.load(full_path)

        return nii.affine, nii.header

    def read_vol_ppdata(self, subject, filename='brainmask', data_format='func1pt8mm'):
        full_path = op.join(self.ppdata_folder,
                            '{subject}', '{data_format}', '{filename}.nii.gz')
        full_path = full_path.format(subject=subject,
                                     data_format=data_format,
                                     filename=filename)
        return nb.load(full_path).get_data()

    def read_betas(self, subject, session_index, trial_index=[], data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage', mask=None):
        data_folder = op.join(self.nsddata_betas_folder,
                              subject, data_format, data_type)
        si_str = str(session_index).zfill(2)

        if type(mask) == np.ndarray:  
            ipf = op.join(data_folder, f'betas_session{si_str}.mat')
            assert op.isfile(ipf), \
                'Error: ' + ipf + ' not available for masking. You may need to download these separately.'
            h5 = h5py.File(ipf, 'r')
            betas = h5.get('betas')
            if len(trial_index) == 0:
                trial_index = slice(0, betas.shape[0])
            return betas[trial_index, np.nonzero(mask)]

        if data_format == 'fsaverage':
            session_betas = []
            for hemi in ['lh', 'rh']:
                hdata = nb.load(op.join(
                    data_folder, f'{hemi}.betas_session{si_str}.mgh')).get_data()
                session_betas.append(hdata)
            out_data = np.squeeze(np.vstack(session_betas))
        else:
            out_data = nb.load(
                op.join(data_folder, f'betas_session{si_str}.nii.gz')).get_fdata()

        if len(trial_index) == 0:
            trial_index = slice(0, out_data.shape[-1])

        return out_data[..., trial_index]

    def read_mapper_results(self, subject, mapper='prf', data_type='angle', data_format='fsaverage'):
        if data_format == 'fsaverage':
            raise NotImplementedError(
                'no mapper results in fsaverage present for now')
        else:  
            return self.read_vol_ppdata(subject=subject, filename=f'{mapper}_{data_type}', data_format=data_format)

    def read_atlas_results(self, subject, atlas='HCP_MMP1', data_format='fsaverage'):
        atlas_name = atlas
        if atlas[:3] in ('rh.', 'lh.'):
            atlas_name = atlas[3:]

        mapp_df = pd.read_csv(os.path.join(self.nsddata_folder, 'freesurfer', 'fsaverage',
                                           'label', f'{atlas_name}.mgz.ctab'), delimiter=' ', header=None, index_col=0)
        atlas_mapping = mapp_df.to_dict()[1]
        atlas_mapping = {y: x for x, y in atlas_mapping.items()}

        if data_format not in ('func1pt8mm', 'func1mm', 'MNI'):
            if atlas[:3] in ('rh.', 'lh.'):  
                ipf = op.join(self.nsddata_folder, 'freesurfer',
                              subject, 'label', f'{atlas}.mgz')
                return np.squeeze(nb.load(ipf).get_data()), atlas_mapping
            else: 
                session_betas = []
                for hemi in ['lh', 'rh']:
                    hdata = nb.load(op.join(
                        self.nsddata_folder, 'freesurfer', subject, 'label', f'{hemi}.{atlas}.mgz')).get_data()
                    session_betas.append(hdata)
                out_data = np.squeeze(np.vstack(session_betas))
                return out_data, atlas_mapping
        else:  
            ipf = op.join(self.ppdata_folder, subject,
                          data_format, 'roi', f'{atlas}.nii.gz')
            return nb.load(ipf).get_fdata(), atlas_mapping

    def list_atlases(self, subject, data_format='fsaverage', abs_paths=False):
        if data_format in ('func1pt8mm', 'func1mm', 'MNI'):
            atlas_files = glob.glob(
                op.join(self.ppdata_folder, subject, data_format, 'roi', '*.nii.gz'))
        else:
            atlas_files = glob.glob(
                op.join(self.nsddata_folder, 'freesurfer', subject, 'label', '*.mgz'))

        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        print('Atlases found in {}:'.format(op.split(atlas_files[0])[0]))
        pp.pprint([op.split(f)[1] for f in atlas_files])
        if abs_paths:
            return atlas_files
        else:  
            return np.unique([op.split(f)[1].replace('lh.', '').replace('rh.', '').replace('.mgz', '').replace('.nii.gz', '') for f in atlas_files])

    def read_behavior(self, subject, session_index, trial_index=[]):

        behavior = pd.read_csv(self.behavior_file.format(subject=subject), delimiter='\t')

        session_behavior = behavior[behavior['SESSION'] == session_index]

        if len(trial_index) == 0:
            trial_index = slice(0, len(session_behavior))

        return session_behavior.iloc[trial_index]

    def read_images(self, image_index, show=False):
        if not hasattr(self, 'stim_descriptions'):
            self.stim_descriptions = pd.read_csv(
                self.stimuli_description_file, index_col=0)

        sf = h5py.File(self.stimuli_file, 'r')
        sdataset = sf.get('imgBrick')
        if show:
            f, ss = plt.subplots(1, len(image_index),
                                 figsize=(6*len(image_index), 6))
            if len(image_index) == 1:
                ss = [ss]
            for s, d in zip(ss, sdataset[image_index]):
                s.axis('off')
                s.imshow(d)
        return sdataset[image_index]

