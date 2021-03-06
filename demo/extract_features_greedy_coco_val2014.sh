# python -m demo.extract_features \
#     --split val2014 \
#     --imgdir /home/tanmayg/Data/gpv/learning_phase_data/coco/images \
#     --outdir /home/tanmayg/Data/bua/

python -m demo.extract_features_greedy \
    --split val2014 \
    --model original_trained \
    --imgdir /home/tanmayg/Data/coco_gpv_split/coco \
    --outdir /home/tanmayg/Data/bua_very_greedy/