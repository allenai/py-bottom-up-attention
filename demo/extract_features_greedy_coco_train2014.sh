# python -m demo.extract_features \
#     --split val2014 \
#     --imgdir /home/tanmayg/Data/gpv/learning_phase_data/coco/images \
#     --outdir /home/tanmayg/Data/bua/

python -m demo.extract_features_greedy \
    --split train2014 \
    --model gpv_trained \
    --imgdir /home/tanmayg/Data/coco_gpv_split/coco \
    --outdir /home/tanmayg/Data/bua_gpv_very_greedy/