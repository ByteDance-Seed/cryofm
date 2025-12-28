# CryoFM2 pretrain config
patch_size = 64
is_debug = False
seed = 42

local_dataset_root = "path/to/cryofm2_pretrain_dataset"

# csv list is paired half maps (map_path1, map_path2), we train our unconditional model on map_path2
pretrain_half_train_pipeline = [
    dict(type="RotCube24", keys=["map_path2"], p=1.0),
    dict(type="MaxValueStandardize", keys=["map_path2"], max_value_key="quantile_max_value2"),
    dict(type="PrecomputedStandardize3D", map_key="map_path2", mean=0.04, std=0.09),
    dict(type="PackInputs", keys=["map_id", "map_path2", "apix"])
]

pretrain_half_test_pipeline = [
    # dict(type="MaxValueStandardize", keys=["map_path2"], max_value_key="quantile_max_value2"),
    dict(type="PackInputs", keys=["map_id", "map_path2", "apix"])
]

split_ratio = 0.7
train_dataset = dict(
    data_info_path=f"{local_dataset_root}/pretrain_map_v1/split/train.csv",
    crop_info_path=f"{local_dataset_root}/pretrain_crop_v1/split/train.csv",
    data_dir=f"{local_dataset_root}/pretrain_map_v1/",
    crop_dir=f"{local_dataset_root}/pretrain_crop_v1/",
    blank_crop_dir=f"{local_dataset_root}/pretrain_blank_crop_v1/",
    data_exclude_path=None,
    patches_per_map=41,
    transform=pretrain_half_train_pipeline,
    split_ratio=split_ratio,
    seed=42)

val_dataset = dict(
        data_info_path=f"{local_dataset_root}/pretrain_map_v1/split/test.csv",
        data_dir=f"{local_dataset_root}/pretrain_map_v1/",
        data_exclude_path=None,
        transform=pretrain_half_test_pipeline,
        seed=42,)

test_dataset = val_dataset

train_dataloader = dict(
    batch_size=12,
    num_workers=24,
    shuffle=True,
    drop_last=True,
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    shuffle=False
)
test_dataloader = val_dataloader

# configuration for model
model_type = "unet"
model = dict(
    sample_size=64,
    in_channels=2, # 1st channel xt, 2nd channel remained for conditional fine-tuning
    out_channels=1,
    time_embedding_type="positional",
    time_embedding_dim=None,
    freq_shift=0,
    flip_sin_to_cos=True,
    down_block_types=("DownBlock3D", "DownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D"),
    up_block_types=("AttnUpBlock3D", "AttnUpBlock3D", "UpBlock3D", "UpBlock3D"),
    block_out_channels=(64, 128, 256, 512),
    layers_per_block=2,
    mid_block_scale_factor=1,
    downsample_padding=1,
    downsample_type="conv",
    upsample_type="conv",
    dropout=0.0,
    act_fn="silu",
    attention_head_dim=8,
    norm_num_groups=32,
    attn_norm_num_groups=None,
    norm_eps=1e-5,
    resnet_time_scale_shift="scale_shift",
)


optimizer = dict(
    lr=1e-4,
    warmup=2000,
)

pl_trainer = dict(
    precision="bf16-mixed",
    num_sanity_val_steps=0,
    check_val_every_n_epoch=None,
    val_check_interval=30,
    max_steps=50,
    devices=1,
    gradient_clip_val=1,
    accumulate_grad_batches=1
)

num_val_samples = 3
keep_last_k = None

ckpt_path = None
resume_path = None
z_crop = None

#validation
mode = "train"

z_scale = dict(
    mean=None,  # 0.04, 0.0
    std=None  # 0.09, 1.0
)

process = "fm"
timestep_sampling = "uniform"
reweight = dict(
    switch=False,
)

ddpm = dict(
    prediction_type="v_prediction",
    cond_drop_threshold=3
)

inference = dict(
    batch_size=16,
    patch_overlap=0
)
