import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import radio
from radio import batchflow as bf
from radio import CTImagesMaskedBatch as CTIMB
from radio.batchflow.research import Research

'''
# read annotation
nodules = pd.read_csv(r'../data/annotations.csv')

# create index and dataset
lunaix = bf.FilesIndex(path=r'../data/s*/*.mhd', no_ext=True)
lunaset = bf.Dataset(index=lunaix, batch_class=CTIMB)

lunaset.split(0.5, shuffle=True)
print(len(lunaset.train))

from radio.pipelines import split_dump

SPACING = (1.7, 1.0, 1.0)  # spacing of scans after spacing unification
SHAPE = (400, 512, 512)  # shape of scans after spacing unification
PADDING = 'reflect'  # 'reflect' padding-mode produces the least amount of artefacts
METHOD = 'pil-simd'  # robust resize-engine

kwargs_default = dict(shape=SHAPE, spacing=SPACING, padding=PADDING, method=METHOD)

crop_pipeline = split_dump(cancer_path=r'../data/lunaset_split/train/cancer',
                           non_cancer_path=r'../data/lunaset_split/train/noncancer',
                           nodules=nodules, fmt='raw', nodule_shape=(32, 64, 64),
                           batch_size=20, **kwargs_default)


(lunaset.train >> crop_pipeline).run()
'''

DIR_CANCER = r'../data/lunaset_split/train/cancer/*'
DIR_NCANCER = r'../data/lunaset_split/train/noncancer/*'


cix = bf.FilesIndex(path=DIR_CANCER, dirs=True)
ncix = bf.FilesIndex(path=DIR_NCANCER, dirs=True)

cancerset = bf.Dataset(index=cix, batch_class=CTIMB)
ncancerset = bf.Dataset(index=ncix, batch_class=CTIMB)

from radio.pipelines import combine_crops
from utils import show_slices # function for plotting batch masks and images

from radio.models import Keras3DUNet, KerasResNoduleNet
from radio.models.keras.losses import dice_loss

unet_config = dict(
    input_shape = (1, 32, 64, 64),
    num_targets = 1,
    loss= dice_loss,
    optimizer='Adam'
)

from radio.batchflow import F, V

train_unet_pipeline = (
    combine_crops(cancerset, ncancerset, batch_sizes=(1, 1))
    .init_variable("iterations", default=0)
    .init_variable("accuracy", init_on_each_run=0)
    .init_variable('current_loss', 0)
    .init_variable("loss_history", init_on_each_run=list)
    .init_model(
        name='3dunet', model_class=KerasResNoduleNet,
        config=unet_config, mode='static'
    )
    .train_model(
        name='3dunet', 
        fetches='loss', 
        save_to=V('current_loss'), 
        mode='w',
        x=F(CTIMB.unpack, component='images', data_format='channels_first'),
        y=F(CTIMB.unpack, component='masks', data_format='channels_first')
    )
    .print("Current loss:")
    .print(V('current_loss'))
    .update_variable('loss_history', value=V('current_loss'), mode='a')
)

train_unet_pipeline.run(batch_size=2, n_epoch=1, bar=True)

loss_history = (
    pd.Series(train_unet_pipeline.get_variable('loss_history'))
    .rolling(32)
    .mean()
    .transform(lambda x: -x)
)

plt.figure(figsize=(7, 5))
loss_history.plot(grid=True)
plt.xlabel('Iteration')
plt.ylabel('dice loss')

plt.savefig(r'../data/plot.png')

keras_unet = train_unet_pipeline.get_model_by_name('3dunet')
keras_unet.save('../data/unet/weights_new.hdf5')