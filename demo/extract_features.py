import os
import io
import h5py
import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
import argparse

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train2014', help='train2014, val2014')
parser.add_argument('--imgdir', type=str, help='path to coco image directory')
parser.add_argument('--outdir', type=str, help='path to save features')
parser.add_argument('--verbose', dest='verbose', action='store_true')

args = parser.parse_args()

def run_detector(raw_image,predictor,num_objects=100,verbose=True):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        if verbose: tqdm.write("Original image size: " + str((raw_height, raw_width)))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        if verbose: tqdm.write("Transformed image size: "+str(image.shape[:2]))
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        if verbose: tqdm.write('Proposal Boxes size: ' + str(proposal.proposal_boxes.tensor.shape))
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        if verbose: tqdm.write('Pooled features size: ' + str(feature_pooled.shape))
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor    
        
        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=num_objects
            )
            if len(ids) == num_objects:
                break
                
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        if verbose: tqdm.write(str(instances))
        
        return instances, roi_features

def main():
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_C4_3x_coco_2014.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    # select whichever model you want to cache features for
    # faster rcnn trained on gpv-train set
    # cfg.MODEL.WEIGHTS = "/home/tanmayg/Data/detectron_output/model_final.pth" 
    # faster rcnn trained on original-train set
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl"
    predictor = DefaultPredictor(cfg)

    imgdir=os.path.join(args.imgdir,args.split) #'/home/tanmayg/Data/gpv/learning_phase_data/coco/images/train2014'
    outdir=args.outdir #'/home/tanmayg/Data/bua/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    feats_h5py = os.path.join(outdir,f'{args.split}.hdf5')
    f = h5py.File(feats_h5py,'w')
    paths = glob.glob(f'{imgdir}/*.jpg')
    for i,path in enumerate(tqdm(paths)):
        im = cv2.imread(path)
        instances, roi_features = run_detector(im,predictor,verbose=args.verbose)
        # instances.pred_boxes: (36,4) x1,y1,x2,y2
        # instances.scores: (36,)
        # instances.classes: (36,)
        # roi_features: (36,2048)
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        feats = roi_features.cpu().numpy()
        im_name = os.path.splitext(os.path.split(path)[1])[0]
        _,subset,im_id = im_name.split('_')
        im_id = int(im_id)
        grp = f.create_group(im_name)
        grp.create_dataset('boxes',data=boxes)
        grp.create_dataset('scores',data=scores)
        grp.create_dataset('classes',data=classes)
        grp.create_dataset('feats',data=feats)

    f.close()


if __name__=='__main__':
    main()