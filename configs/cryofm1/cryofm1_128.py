# CryoFM1 (S)Small input shape 128, 3Apix, model architecture p4 d3
patch_size = 128

train_pipeline = [
    dict(type="LoadMapFromFile"),
    dict(type="CenterCrop3D", patch_size=patch_size),
    dict(type="ClipNorm3D", min_value=0, max_percentile=99.999),
    dict(type="RandomRotation3D", keys=["map_data"], p=1.0, order=1),
    dict(type="ClipNorm3D", min_value=0, max_percentile=99.999),
    dict(type="Pad3D", keys=["map_data"], size=patch_size),
    dict(type="RandomPatches3D", patch_size=patch_size, num_patches=1),
    dict(type="PackInputs", keys=["map_id", "src_shape", "patches"])
]

test_pipeline = [
    dict(type="LoadMapFromFile"),
    dict(type="CenterCrop3D", patch_size=patch_size),
    dict(type="ClipNorm3D", min_value=0, max_percentile=99.999),
    dict(type="Pad3D", keys=["map_data"], size=patch_size),
    dict(type="PackInputs", keys=["map_id", "map_data", "src_shape", ])
]

train_dataset = dict(
        type='EMDBMapDataset',
        data_root='path/to/cryofm1_3apix_dataset/',
        ann_file='split/train.csv',
        pipeline=train_pipeline)

val_dataset = dict(
        type='EMDBMapDataset',
        data_root='path/to/cryofm1_3apix_dataset/',
        ann_file='split/test.csv',
        pipeline=test_pipeline)

test_dataset = val_dataset

train_dataloader = dict(
    batch_size=16,
    num_workers=12,
    shuffle=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    shuffle=False
)

test_dataloader = val_dataloader

# ======

seed = 42

hdit_model = dict(
    type="image_transformer_v2",
    input_channels=1,
    input_size=[patch_size, patch_size, patch_size],  # dummy
    patch_size=[4, 4, 4],
    depths=[2, 2, 12],
    widths=[320, 640, 1280],
    self_attns=[
        {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
        {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
        {"type": "global", "d_head": 64}
    ]
)

model_type = "hdit"

optimizer = dict(
    lr=1e-4,
    warmup=2000,
)

pl_trainer = dict(
    # max_epochs=200,
    # check_val_every_n_epoch=5,
    precision="bf16",
    num_sanity_val_steps=0,
    check_val_every_n_epoch=None,
    val_check_interval=10000,
    max_steps=300000,
    devices=1,
    gradient_clip_val=1.,
)

# extra_train_args = dict(
#     split_batch_size=8,
#     split_batch_grad_acc=1,
# )

num_val_samples = 3
keep_last_k = None

ckpt_path = None
z_crop = None

mode = "train"

z_scale = dict(
    mean=0.04,  # 0.04, 0.0
    std=0.09  # 0.09, 1.0
)

process = "fm"
timestep_sampling = "logit-normal"
reweight = dict(
    switch=False,
)

ddpm = dict(
    prediction_type="v_prediction"
)
