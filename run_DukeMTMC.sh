###

CUDA_VISIBLE_DEVICES=2	 python train.py --config_file='configs/softmax_triplet_with_center.yml' --dataset="dukemtmcreid" \
    MODEL.DEVICE_ID "('2')" \
    DATASETS.NAMES "('dukemtmc')" \
    OUTPUT_DIR "('./results/dukemtmc/train')" \
    MODEL.PREMODEL "('./results/dukemtmc/resnet50_baseline_model.pth')" \SOLVER.all_MAX_EPOCHS 200 \
    SOLVER.all_EVAL_PERIOD [100,200] \
    DATALOADER.NUM_INSTANCE 4 \
    SOLVER.IMS_PER_BATCH 64 \
    TEST.IMS_PER_BATCH 128 \
    SOLVER.edge_k 3 \
    SOLVER.all_part True \
    TEST.WEIGHT "('./results/dukemtmc/train/resnet50_model_200.pth')"
    



