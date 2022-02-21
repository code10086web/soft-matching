###

CUDA_VISIBLE_DEVICES=6	 python train.py --config_file='configs/softmax_triplet_with_center.yml' --dataset="cuhk03" --cuhk03-labeled \
    MODEL.DEVICE_ID "('6')" \
    DATASETS.NAMES "('cuhk03')" \
    OUTPUT_DIR "('./results/cuhk03/train_labeled')" \
    MODEL.PREMODEL "('./results/cuhk03/labeled_resnet50_baseline_model.pth')" \
    SOLVER.all_MAX_EPOCHS 10 \
    SOLVER.all_EVAL_PERIOD [1,2,3,4,5,6,7,8,9,10] \
    DATALOADER.NUM_INSTANCE 4 \
    SOLVER.IMS_PER_BATCH 64 \
    TEST.IMS_PER_BATCH 128 \
    SOLVER.edge_k 3 \
    SOLVER.all_part True
    



