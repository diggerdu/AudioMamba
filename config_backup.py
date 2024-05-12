# for model training

exp_name = "exp_test"
workspace = "/home/tiger/DB/knut/tscam_sed"
dataset_path = "/home/tiger/DB/knut/data/audioset"
index_type = "full_train"

loss_type = "clip_bce" # "asl_loss" # "clip_bce"
balanced_data = True

resume_checkpoint = "saved_model_st_768_pretrain.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_st_2048/saved_model_3.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_tscam_test/checkpoint/lightning_logs/version_11/checkpoints/l-epoch=0-mAP=0.453-mAUC=0.976.ckpt"
# "saved_model_1.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_49/checkpoints/l-epoch=40-mAP=0.445-mAUC=0.964.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin/checkpoint/lightning_logs/version_15/checkpoints/l-epoch=22-mAP=0.470-mAUC=0.962.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_13/checkpoints/l-epoch=14-mAP=0.466-mAUC=0.959.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_3/checkpoints/l-epoch=21-mAP=0.466-mAUC=0.958.ckpt"

# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_49/checkpoints/l-epoch=40-mAP=0.445-mAUC=0.964.ckpt"

# "/home/tiger/DB/knut/tscam_sed/results/exp_pann/checkpoint/lightning_logs/version_5/checkpoints/l-epoch=10-mAP=0.428-mAUC=0.970.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_12/checkpoints/l-epoch=13-mAP=0.431-mAUC=0.958.ckpt"

class_map_path = "class_hier_map.npy"
class_filter = None 
# [516, 417, 152, 87, 3, 35, 449, 387, 355, 30, 157, 501, 242, 285, 302, 96, 339, 245, 63, 346, 445, 457, 380, 145, 300, 337, 105, 23, 420, 109, 134, 514, 231, 151, 139, 384, 34, 22, 290, 238, 371, 196, 164, 188, 166, 67, 378, 61, 335, 93, 429, 187, 517, 436, 190, 189, 261, 54, 288, 161, 215, 97, 29, 168, 385, 340, 83, 173, 150, 394, 110, 218, 492, 255, 198, 458, 301, 291, 148, 467, 293, 249, 336, 214, 359, 165, 95, 127, 463, 162, 136, 171, 403, 170, 422, 86, 393, 392, 311, 141, 
# 312, 72, 465, 309, 386, 167, 64, 310, 85, 382, 366, 377, 56, 201, 462, 73, 17, 140, 107, 415, 204, 347, 118, 92, 197, 172, 6, 71, 117, 383, 106, 432, 174, 334, 202, 372, 121, 100, 342, 186, 491, 59, 193, 68, 203, 128, 74, 28, 476, 169, 194, 58, 69, 281, 177, 131, 490, 205
# , 331, 367, 390, 324, 428, 289, 427, 320, 199, 192, 81, 369, 318, 399, 40, 155, 472, 57, 287, 430, 323, 60, 89, 115, 15, 206, 400, 411, 126, 423, 116, 195, 455, 332, 77, 328, 329, 160, 414, 147, 119, 185, 325, 286, 91, 183, 82, 90, 99, 333, 43, 431, 0, 374, 498, 137, 98
# , 373, 209, 129, 132, 133, 120, 208, 494, 211, 12, 397, 101, 212, 103, 102, 207, 396, 322, 210]

#[434, 504, 475, 447, 130, 486, 510, 280, 507, 513, 502, 444, 38, 299, 235, 270, 349, 500, 485, 1, 10, 505, 279, 237, 135, 473, 509,
#  306, 488, 266, 508, 5, 273, 466, 260, 297, 487, 233, 277, 452, 520, 327, 440, 226, 348, 407, 9, 269, 2, 275, 32, 51, 489, 503, 239,
#  124, 216, 450, 442, 219, 284, 159, 523, 221, 46, 158, 113, 41, 8, 227, 264, 518, 33, 405, 276, 375, 506, 298, 258, 358, 80, 471, 
#  341, 25, 225, 44, 519, 125, 356, 55, 252, 409, 354, 352, 404, 410, 497, 483, 200, 448, 232, 524, 178, 477, 11, 230, 248, 18, 388, 4,
#  26, 265, 338, 229, 163, 357, 345, 217, 234, 413, 412, 268, 274, 454, 254, 246, 114, 398, 4, 456, 468, 526, 350, 512, 389, 326, 314,
#  21, 304, 313, 267, 460, 439, 94, 222, 451, 278, 42, 19, 317, 53, 240, 353, 459, 305, 251, 220, 424, 253, 478, 441, 76, 522, 108, 2,
#  6, 24, 511, 364, 176, 122, 495, 376, 224, 70, 461, 496, 14]

model_type = "swin" # ["pann","vit","swin"]



debug = False

random_seed = 970131 # 19970318 970131 12412 127777 1009
batch_size = 16 * 8 # 128
learning_rate = 1e-3
max_epoch = 100
num_workers = 4

lr_scheduler_epoch = [10,20,30] # 【10, 20, 30] [10,15,100] [5,10,15]
lr_rate = [5e-2, 0.1, 0.2] # [5e-2, 0.1, 0.4] [5e-2, 0.1, 0.2] [0.02, 0.05, 0.1]

enable_tscam = True # tscam
enable_token_label = False # deprecated 
enable_time_shift = False # shift time
enable_label_enhance = False # enhance hierarchical label
enable_repeat_mode = False # repeat the spectrogram / reshape the spectrogram
token_label_range = [0.2,0.6]

# for signal processing
sample_rate = 32000
clip_samples = sample_rate * 10 # audio_set 10-sec clip
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)
# for data collection
classes_num = 527
patch_size = (25, 4)
crop_size = None # int(clip_samples * 0.5)

# for swin hyperparamter
swin_window_size = 8 # 12 # 8
swin_img_size =  256 # 384 # 256
swin_patch_size = 4 #4
swin_stride = (4, 4)
swin_num_head = [4,8,16,32] # [3,6,12,24]
swin_dim = 96 # 128 # 96  # 256
swin_depth = [2,2,6,2] # 2 2 18 2 # 2,2,6,2] 
swin_use_max = False
swin_pretrain_path = None
#"pretrain/swin_tiny_c24_patch4_window8_256.pth"
# "pretrain/swin_tiny_c24_patch4_window8_256.pth"
# "pretrain/swin_tiny_c24_patch4_window8_256.pth"
# "pretrain/swin_tiny_c24_patch4_window8_256.pth"
# "pretrain/swin_tiny_c24_patch4_window8_256.pth"
# "pretrain/moco_swin_tiny_p4_w8_256.pth"
# "pretrain/swin_base_patch4_window12_384.pth"
# "pretrain/swin_base_patch4_window12_384.pth"
# "pretrain/moco_swin_tiny_p4_w8_256.pth"
swin_attn_heatmap = False

# Change: Test Multi-Swin
swin_hier_output = False # True

# for test

heatmap_dir = "/home/tiger/DB/knut/tscam_sed/heatmap_output"
test_file = "tscam_ts_2048_attn"


retrieval_index = [15382, 9202, 130, 17618, 17157, 17516, 16356, 6165, 13992, 9238, 5550, 5733, 1914, 1600, 3450, 13735, 11108, 3762, 
    9840, 11318, 8131, 4429, 16748, 4992, 16783, 12691, 4945, 8779, 2805, 9418, 2797, 14357, 5603, 212, 3852, 12666, 1338, 10269, 2388, 8260, 4293, 14454, 7677, 11253, 5060, 14938, 8840, 4542, 2627, 16336, 8992, 15496, 11140, 446, 6126, 10691, 8624, 10127, 9068, 16710, 10155, 14358, 7567, 5695, 2354, 8057, 17635, 133, 16183, 14535, 7248, 4560, 14429, 2463, 10773, 113, 2462, 9223, 4929, 14274, 4716, 17307, 4617, 2132, 11083, 1039, 1403, 9621, 13936, 2229, 2875, 17840, 9359, 13311, 9790, 13288, 4750, 17052, 8260, 14900]


# for ensemble test 

ensemble_checkpoints = []
ensemble_strides = []

# pann ensemble
# "/home/tiger/DB/knut/tscam_sed/results/exp_pann/checkpoint/lightning_logs/version_5/checkpoints/l-epoch=10-mAP=0.428-mAUC=0.970.ckpt"


# swin ensemble
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin/checkpoint/lightning_logs/version_15/checkpoints/l-epoch=22-mAP=0.470-mAUC=0.962.ckpt"


# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_12/checkpoints/l-epoch=13-mAP=0.431-mAUC=0.958"
# swin_tscam ensemble
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_15/checkpoints/l-epoch=24-mAP=0.473-mAUC=0.963.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_16/checkpoints/l-epoch=33-mAP=0.474-mAUC=0.961.ckpt"
# "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_17/checkpoints/l-epoch=23-mAP=0.474-mAUC=0.964.ckpt"
# wa_model_path = "saved_model_st_768_pretrain.ckpt"
wa_model_path = "/mnt/bd/duxingjianhl4t/knut/tscam_sed/results/exp_st_768_pretrain/checkpoint/lightning_logs/version_3/wa.ckpt"

# esm_model_folder = "/home/tiger/DB/knut/tscam_sed/results/exp_st_768_pretrain/checkpoint/lightning_logs/version_2/checkpoints/"

esm_model_folder = "/mnt/bd/duxingjianhl4t/knut/tscam_sed/results/exp_st_768_pretrain/checkpoint/lightning_logs/version_3/checkpoints"

esm_model_pathes = [
    "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_15/checkpoints/l-epoch=24-mAP=0.473-mAUC=0.963.ckpt",
    "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_16/checkpoints/l-epoch=33-mAP=0.474-mAUC=0.961.ckpt",
    "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_17/checkpoints/l-epoch=23-mAP=0.474-mAUC=0.964.ckpt",
    "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_49/checkpoints/l-epoch=40-mAP=0.445-mAUC=0.964.ckpt",
    "/home/tiger/DB/knut/tscam_sed/results/exp_swin_tscam/checkpoint/lightning_logs/version_27/checkpoints/l-epoch=22-mAP=0.442-mAUC=0.966.ckpt",
]

# for framewise localization
fl_local = False # indicate if we need to use this dataset for the framewise detection
fl_dataset = "/home/tiger/DB/knut/data/desed_sed/dataset/eval.npy"  
fl_class_num = [
    "Speech", "Frying", "Dishes", "Running_water",
    "Blender", "Electric_shaver_toothbrush", "Alarm_bell_ringing",
    "Cat", "Dog", "Vacuum_cleaner"
]

fl_audioset_mapping = [
    [0,1,2,3,4,5,6,7],
    [366, 367, 368],
    [364],
    [288, 289, 290, 291, 292, 293, 294, 295, 296, 297],
    [369],
    [382],
    [310, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402],
    [81, 82, 83, 84, 85],
    [74, 75, 76, 77, 78, 79],
    [377]
]
