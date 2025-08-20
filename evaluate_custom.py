import argparse
import numpy as np
import os
import torch
import pdb
from tqdm import tqdm

from utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev,iou3d,iou3d_custom
from dataset import Kitti, get_dataloader
from model import PointPillars

import numpy as np
from shapely.geometry import Polygon
import pandas as pd
def get_bev_polygon(box):
    """Returns a 2D polygon for BEV projection."""
    x, y, _, w, l, _, theta = box
    dx, dy = w / 2, l / 2
    corners = np.array([
        [dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]
    ])
    # Rotate and translate
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_corners = (R @ corners.T).T + np.array([x, y])
    return Polygon(rotated_corners)

def compute_3d_iou_matrix(bboxes1, bboxes2):
    """
    Computes the IoU matrix between two sets of 3D bounding boxes.
    
    Args:
        bboxes1: (n, 7) array of bounding boxes (x, y, z, w, l, h, theta)
        bboxes2: (m, 7) array of bounding boxes
    
    Returns:
        iou_matrix: (n, m) IoU matrix
    """
    n, m = len(bboxes1), len(bboxes2)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        poly1 = get_bev_polygon(bboxes1[i])
        z1, h1 = bboxes1[i][2], bboxes1[i][5]
        z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2

        for j in range(m):
            poly2 = get_bev_polygon(bboxes2[j])
            z2, h2 = bboxes2[j][2], bboxes2[j][5]
            z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2

            # Compute BEV intersection area
            if not poly1.intersects(poly2):
                continue
            intersection_area = poly1.intersection(poly2).area

            # Compute height overlap
            h_int = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

            # Compute volumes
            intersection_volume = intersection_area * h_int
            vol1 = bboxes1[i][3] * bboxes1[i][4] * bboxes1[i][5]
            vol2 = bboxes2[j][3] * bboxes2[j][4] * bboxes2[j][5]
            union_volume = vol1 + vol2 - intersection_volume

            # Compute IoU
            iou_matrix[i, j] = intersection_volume / union_volume if union_volume > 0 else 0.0

    return iou_matrix


def compute_tp_fp_fn(iou_matrix, iou_threshold=0.5):
    """
    Computes True Positives (TP), False Positives (FP), and False Negatives (FN).

    Args:
        iou_matrix: (n, m) IoU values (GT vs. detections)
        iou_threshold: float, IoU threshold for matching

    Returns:
        TP: int, True Positives
        FP: int, False Positives
        FN: int, False Negatives
    """
    n, m = iou_matrix.shape  # n: ground truths, m: detections

    matches = iou_matrix >= iou_threshold  # Valid matches
    gt_matched = np.zeros(n, dtype=bool)
    det_matched = np.zeros(m, dtype=bool)

    for j in range(m):
        best_gt = np.argmax(iou_matrix[:, j])
        if matches[best_gt, j] and not gt_matched[best_gt]:
            gt_matched[best_gt] = True
            det_matched[j] = True

    TP = np.sum(gt_matched)
    FP = np.sum(~det_matched)
    FN = np.sum(~gt_matched)
    
    return TP, FP, FN

def compute_precision_recall(tp, fp, fn):
    """
    Computes precision and recall values.

    Returns:
        precision: float
        recall: float
    """
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    return precision, recall

def compute_average_precision(recall, precision):
    """
    Computes Average Precision using 11-point interpolation.

    Args:
        recall: np.array of recall values
        precision: np.array of precision values

    Returns:
        ap: float, Average Precision
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):  # 11 recall points
        p = np.max(precision[recall >= t]) if np.any(recall >= t) else 0
        ap += p
    return ap / 11

def compute_class_ap(gt_bboxes3d, gt_cls, det_bboxes3d, det_cls, det_score, class_name, iou_threshold=0.5):
    """
    Computes AP for a specific class.

    Args:
        gt_bboxes3d: (n, 7) GT boxes
        gt_cls: List of GT class labels
        det_bboxes3d: (m, 7) Detected boxes
        det_cls: List of detection class labels
        det_score: List of detection scores
        class_name: Target class ("car", "pedestrian", "cyclist")
        iou_threshold: IoU threshold for TP matching

    Returns:
        AP: Average Precision for the class
    """
    
    # Filter by class
    gt_bboxes = gt_bboxes3d[np.array(gt_cls) == class_name]
    det_bboxes = det_bboxes3d[np.array(det_cls) == class_name]
    det_scores = det_score[np.array(det_cls) == class_name]

    if len(gt_bboxes) == 0:
        return 0.0  # No GT means AP is 0

    # Sort detections by confidence
    sorted_indices = np.argsort(-det_scores)
    det_bboxes = det_bboxes[sorted_indices]

    # Compute IoU
    iou_matrix = compute_3d_iou_matrix(gt_bboxes, det_bboxes)

    # Compute TP, FP, FN
    tp, fp, fn = compute_tp_fp_fn(iou_matrix, iou_threshold)

    # Compute Precision-Recall curve
    precision, recall = compute_precision_recall(tp, fp, fn)

    # Compute AP
    ap = compute_average_precision(recall, precision)
    return ap




def compute_tp_fp_fn(iou_matrix, conf_scores, det_classes, gt_classes, iou_threshold=0.5):
    n, m = iou_matrix.shape  # IoU matrix dimensions (n: ground truths, m: detections)
    assigned_gt = np.full(m, -1)  # Stores assigned ground truth index for each detection
    gt_matched = np.zeros(n, dtype=bool)  # Tracks if a ground truth has been matched
    results = []

    # Step 1: Match detections to ground truths
    for j in range(m):
        best_gt_idx = np.argmax(iou_matrix[:, j])
        if iou_matrix[best_gt_idx, j] > iou_threshold and gt_classes[best_gt_idx] == det_classes[j]:
            assigned_gt[j] = best_gt_idx  # Assign detection to this GT
            
            # First valid detection for GT is TP, others (duplicates) are FP
            is_tp = not gt_matched[best_gt_idx]
            gt_matched[best_gt_idx] = True
        else:
            is_tp = False  # No valid GT match -> FP

        # Store TP/FP results
        results.append({
            "Confidence": conf_scores[j],
            "Detection Class": det_classes[j],
            "Matched GT Class": gt_classes[assigned_gt[j]] if assigned_gt[j] != -1 else "None",
            "TP/FP": "TP" if is_tp else "FP"
        })

    # Step 2: Count FN (ground truths that were never matched)
    fn_count = np.sum(~gt_matched)

    # Step 3: Sort detections by confidence score (descending)
    results.sort(reverse=True, key=lambda x: x["Confidence"])
    return results
    """ return {
        "detections": results,
        "FN": fn_count  # Number of ground truths that were not matched
    } """

def add_tp_fp_column(df,gt_sum):
    df['tp'] = 0
    df['fp'] = 0

    df.loc[df['TP/FP'] == 'TP', 'tp'] += 1
    df.loc[df['TP/FP'] == 'FP', 'fp'] += 1
    

    # Compute accumulated TP and FP
    df['acc tp'] = df['tp'].cumsum()
    df['acc fp'] = df['fp'].cumsum()

    # Compute Precision and Recall
    df['precesion'] = df['acc tp'] / (df['acc tp'] + df['acc fp'])
    df['recall'] = df['acc tp'] / gt_sum

    return df




def avg_precesion(df_input , class_name):
    df = df_input[['recall','precesion']] 
    ### **1️⃣ All-Point Interpolation (Trapezoidal AUC)**
    # Sort recall values in ascending order
    df = df.sort_values(by='recall', ascending=True)

    # Ensure precision is non-increasing (right to left)
    df['precesion'] = df[::-1]['precesion'].cummax()

    # Compute AP using the trapezoidal rule
    ap_all_points = np.sum(np.diff(df['recall'], prepend=0) * df['precesion'])

    ### **2️⃣ 11-Point Interpolation Method**
    # Standard recall levels: [0.0, 0.1, ..., 1.0]
    recall_levels = np.linspace(0.0, 1.0, 11)

    # Get max precision at or above each recall level
    precision_at_levels = [df.loc[df['recall'] >= r, 'precesion'].max() if (df['recall'] >= r).any() else 0 for r in recall_levels]

    # Compute AP as the mean of these 11 precision values
    ap_11_point = np.mean(precision_at_levels)

    ### **Print Results**
    print(f"AP (All-Point Interpolation): {ap_all_points * 100:.2f}%",class_name )
    #print(f"AP (11-Point Interpolation): {ap_11_point * 100:.2f}%",class_name)




def do_eval(det_results, gt_results, CLASSES, saved_path):
    '''
    det_results: list,
    gt_results: dict(id -> det_results)
    CLASSES: dict
    '''
    assert len(det_results) == len(gt_results)
    #f = open(os.path.join(saved_path, 'eval_results.txt'), 'w')
    df = pd.DataFrame(columns=['confidence', 'detection Class', 'matched gt class', 'TP/FP'])
    ids = list(sorted(gt_results.keys()))
    total_car = []
    total_pedestrian = []
    total_cyclist = [] 
    for id in ids:
        if(det_results[id]['score'].any() > 0.1 ):
            gt_result = gt_results[id]['annos']
            det_result = det_results[id]

            # 1.2, bev iou
            gt_location = gt_result['location'].astype(np.float32)
            gt_dimensions = gt_result['dimensions'].astype(np.float32)
            gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
            det_location = det_result['location'].astype(np.float32)
            det_dimensions = det_result['dimensions'].astype(np.float32)
            det_rotation_y = det_result['rotation_y'].astype(np.float32)
            det_score = (det_result['score'].astype(np.float32))
            det_cls = (det_result['name'])
            gt_cls = (gt_result['name'])
            gt_bboxes3d = (np.concatenate([gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1))
            det_bboxes3d = (np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1))
            total_car.append(gt_cls.tolist().count('car'))
            total_pedestrian.append(gt_cls.tolist().count('pedestrian'))
            total_cyclist.append(gt_cls.tolist().count('cyclist'))
            iou_matrix = compute_3d_iou_matrix(gt_bboxes3d, det_bboxes3d)
            tp_fp_results = compute_tp_fp_fn(iou_matrix, det_score, det_cls, gt_cls)
            for i in range(len(tp_fp_results)):
                #print(tp_fp_results[i])
                new_row = {'confidence': tp_fp_results[i]['Confidence'], 'detection Class': tp_fp_results[i]['Detection Class'], 'matched gt class': tp_fp_results[i]['Matched GT Class'], 'TP/FP': tp_fp_results[i]['TP/FP']}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df[['tp', 'fp', 'acc tp', 'acc fp', 'precesion', 'recall']] = None 
    df_cyclist = df[df['detection Class'] == 'cyclist']
    df_car = df[df['detection Class'] == 'car']
    df_pedestrian = df[df['detection Class'] == 'pedestrian']
    df_car = df_car.sort_values(by='confidence', ascending=False)
    df_pedestrian = df_pedestrian.sort_values(by='confidence', ascending=False)
    df_cyclist = df_cyclist.sort_values(by='confidence', ascending=False)
    df_car = add_tp_fp_column(df_car,sum(total_car))
    df_pedestrian = add_tp_fp_column(df_pedestrian,sum(total_pedestrian))
    df_cyclist = add_tp_fp_column(df_cyclist,sum(total_cyclist))
    avg_precesion(df_car,"car")
    avg_precesion(df_pedestrian,"pedestrian")
    avg_precesion(df_cyclist,"cyclist")
    df_cyclist.to_csv("df_cyclist.csv", index=False)  # Saves without the index
    df_car.to_csv("df_car.csv", index=False)  # Saves without the index
    df_pedestrian.to_csv("df_pedestrian.csv", index=False)  # Saves without the index
    df.to_csv("dataframe.csv", index=False)  # Saves without the index
        
    

def main(args):
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    CLASSES = Kitti.CLASSES
    #print(CLASSES)
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    #print(LABEL2CLASSES)
    if not args.no_cuda:
        model = PointPillars(nclasses=args.nclasses).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=args.nclasses)
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    saved_submit_path = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit_path, exist_ok=True)

    pcd_limit_range = np.array([0, -40, -5, 70, 40, 1], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        format_results = {}
        print('Predicting and Formatting the results.')
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            batch_results = model(batched_pts=batched_pts, 
                                  mode='val',
                                  batched_gt_bboxes=batched_gt_bboxes, 
                                  batched_gt_labels=batched_labels)
            # pdb.set_trace()
            for j, result in enumerate(batch_results):
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                

                idx =  [int(str(item.split('/')[-1].split('.')[0])[-4:])  for item in data_dict['batched_velodyne']][0] 

                result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)

                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'] , result_filter['scores']

                for lidar_bbox, label, score in \
                    zip(lidar_bboxes, labels, scores):
                    
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    alpha = 0
                    format_result['alpha'].append(alpha)
                    format_result['bbox'].append(lidar_bbox)
                    format_result['dimensions'].append(lidar_bbox[3:6])
                    format_result['location'].append(lidar_bbox[:3])
                    format_result['rotation_y'].append(lidar_bbox[6])
                    format_result['score'].append(score)
                
                write_label(format_result, os.path.join(saved_submit_path, f'{str(idx)}.txt'))

                format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
        print(saved_path)
        write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))
    
    print('Evaluating.. Please wait several seconds.')
    print(saved_path)
    do_eval(format_results, val_dataset.data_infos, CLASSES, saved_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--saved_path', default='results', help='your saved path for predicted results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
