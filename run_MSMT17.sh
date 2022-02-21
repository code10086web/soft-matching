###

CUDA_VISIBLE_DEVICES=7	 python train.py --config_file='configs/softmax_triplet_with_center.yml' --dataset="msmt17" \
    MODEL.DEVICE_ID "('7')" \
    DATASETS.NAMES "('msmt17')" \
    OUTPUT_DIR "('./results/msmt17/train')" \
    MODEL.PREMODEL "('./results/msmt17/resnet50_baseline_model.pth')" \
    SOLVER.all_MAX_EPOCHS 200 \
    SOLVER.all_EVAL_PERIOD [200] \
    DATALOADER.NUM_INSTANCE 4 \
    SOLVER.IMS_PER_BATCH 64 \
    TEST.IMS_PER_BATCH 128 \
    SOLVER.edge_k 3 \
    SOLVER.all_part True 
    



