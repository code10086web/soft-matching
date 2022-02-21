###

CUDA_VISIBLE_DEVICES=5  python test.py --config_file='configs/softmax_triplet_with_center.yml' --dataset="msmt17" \
    MODEL.DEVICE_ID "('5')" \
    MODEL.NAME "('resnet50_ibn_a')" \
    MODEL.PRETRAIN_PATH "('/home/Teacao/Documents/code-normal/r50_ibn_a.pth')" \
    MODEL.PREMODEL "('/home/Teacao/Documents/2019-CVPRW-BagofTricks/results/msmt17/train/resnet50_ibn_a_model_120.pth')" \
    DATASETS.NAMES "('msmt17')" \
    OUTPUT_DIR "('./results/msmt17/test')" \
    SOLVER.all_MAX_EPOCHS 200 \
    SOLVER.all_EVAL_PERIOD [200] \
    DATALOADER.NUM_INSTANCE 4 \
    SOLVER.IMS_PER_BATCH 64 \
    TEST.IMS_PER_BATCH 128 \
    SOLVER.edge_k 3 \
    SOLVER.CHECKPOINT_PERIOD 200 \
    SOLVER.all_part True \
    TEST.WEIGHT "('./results/msmt17/train/resnet50_ibn_a_model_200.pth')"

    



