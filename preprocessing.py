import os
from glob import glob
import shutil
from tqdm import tqdm
# import dicom2nifti
import numpy as np
import nibabel as nib
from monai.transforms import (
    # AddChanneld,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    Flipd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ConcatItemsd,
    ScaleIntensityd,
    AdjustContrastd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    # LoadNiftid,
    # AddChanneld,

)
from monai.data import DataLoader, Dataset, CacheDataset,decollate_batch
from monai.utils import set_determinism
from monai.utils import first
import matplotlib.pyplot as plt
import torch
import monai
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
from monai.utils import first
import matplotlib.pyplot as plt
import torch
import monai

import os
from glob import glob
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import Compose

def prepare(in_dir, cache=False):
    set_determinism(seed=0)

    path_train_volumes_t2w = sorted(glob(os.path.join(in_dir, "TrainVolumes", "t2w", "*.nii.gz")))
    path_train_volumes_adc = sorted(glob(os.path.join(in_dir, "TrainVolumes", "adc", "*.nii.gz")))
    path_train_volumes_hbv = sorted(glob(os.path.join(in_dir, "TrainVolumes", "hbv", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "Binary_TrainSegmentation", "*.nii.gz")))

    path_test_volumes_adc = sorted(glob(os.path.join(in_dir, "ValidVolumes", "adc", "*.nii.gz")))
    path_test_volumes_hbv = sorted(glob(os.path.join(in_dir, "ValidVolumes", "hbv", "*.nii.gz")))
    path_test_volumes_t2w = sorted(glob(os.path.join(in_dir, "ValidVolumes", "t2w", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "Binary_ValidSegmentation", "*.nii.gz")))

    # Extract patient IDs
    def extract_patient_id(file_path):
        filename = os.path.basename(file_path)
        patient_id = filename.split("_")[0]
        return patient_id

    train_patient_ids = [extract_patient_id(path) for path in path_train_volumes_t2w]
    test_patient_ids = [extract_patient_id(path) for path in path_test_volumes_t2w]

    train_files = [{"patient_id": pid, "t2w": t2w_image, "adc": adc_image, "hbv": hbv_image, "seg": label_name}
                    for pid, t2w_image, adc_image, hbv_image, label_name
                    in zip(train_patient_ids, path_train_volumes_t2w, path_train_volumes_adc, path_train_volumes_hbv, path_train_segmentation)]

    test_files = [{"patient_id": pid, "t2w": t2w_image, "adc": adc_image, "hbv": hbv_image, "seg": label_name}
                    for pid, t2w_image, adc_image, hbv_image, label_name
                    in zip(test_patient_ids, path_test_volumes_t2w, path_test_volumes_adc, path_test_volumes_hbv, path_test_segmentation)]

    train_transforms = Compose(
    [
    LoadImaged(keys=["t2w", "adc","hbv", "seg"]),
    EnsureChannelFirstd(keys=["t2w", "adc","hbv","seg"]), # I suppose you seg data already have 2 channels
    ConcatItemsd(keys=["t2w", "adc","hbv"], name="img"),
    # Lambda(func=lambda x: x.permute(0, 4, 1, 2, 3)),  # Reordering to (B, C, H, W, D)
    Resized(keys=["img", "seg"], spatial_size=(160,128,24)),
    # Spacingd( keys=["img", "seg"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
    
    RandAffined(keys=['img', 'seg'], prob=0.2, translate_range=10.0),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    
    RandFlipd(keys=["img", "seg"],spatial_axis=[0],prob=0.50),
    RandFlipd(keys=["img", "seg"],spatial_axis=[1],prob=0.50,),
    RandFlipd(keys=["img", "seg"],spatial_axis=[2],prob=0.50,),
   

    RandGaussianNoised(keys='img', prob=0.4),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="img", factors=0.1, prob=0.4),
    RandShiftIntensityd(keys="img", offsets=0.1, prob=0.4),
    # ScaleIntensityd(minv=0, maxv=450,keys="img"),
    # AdjustContrastd(keys="img",gamma=2.0),

    ToTensord(keys=["img", "seg"]),
    ]
    )


    test_transforms = Compose(
    [
    LoadImaged(keys=["t2w", "adc","hbv","seg"]),
    EnsureChannelFirstd(keys=["t2w", "adc","hbv","seg"]),
    ConcatItemsd(keys=["t2w", "adc","hbv"], name="img"),
    # Lambda(func=lambda x: x.permute(0, 4, 1, 2, 3)),  # Reordering to (B, C, H, W, D)
    Resized(keys=["img", "seg"], spatial_size=(160,128,24)),
    # Spacingd( keys=["img", "seg"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
    
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    # ScaleIntensityd(minv=0, maxv=450,keys="img"),
    # AdjustContrastd(keys="img",gamma=2.0),
    
    ToTensord(keys=["img", "seg"]),
    ]
    )
    
    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader


