###

CUDA_VISIBLE_DEVICES=1	 python train.py --config_file='configs/softmax_triplet_with_center.yml' --dataset="cuhk03" \
    MODEL.DEVICE_ID "('1')" \
    DATASETS.NAMES "('cuhk03')" \
    OUTPUT_DIR "('./results/cuhk03/train_detected')" \
    MODEL.PREMODEL "('./results/cuhk03/detected_resnet50_baseline_model.pth')" \
    SOLVER.all_MAX_EPOCHS 200 \
    SOLVER.all_EVAL_PERIOD [1,5,100,120,200] \
    DATALOADER.NUM_INSTANCE 4 \
    SOLVER.IMS_PER_BATCH 64 \
    TEST.IMS_PER_BATCH 128 \
    SOLVER.edge_k 3 \
    SOLVER.all_part True \
    TEST.WEIGHT "('./results/cuhk03/train_detected/resnet50_model_200.pth')"
    



