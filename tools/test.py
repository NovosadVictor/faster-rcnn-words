from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" Test images """

#my imports
import sys
import xml.etree.ElementTree as ET

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16


CLASSES = ('__background__',
           'aeroexpress', 'microsoft', 'apple', 'google', )

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_{}.ckpt',)}
DATASETS= {'my_dataset': ('my_dataset_train', )}


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if (xB - xA) <= 0 or (yB - yA) <= 0:
        return 0
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def load_annotations(xml_name):
    with open(os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'Annotations', xml_name), 'r') as f:
        tree = ET.parse(f)
        objs = tree.findall('object')
        num_objs = len(objs)

        gt_boxes = np.zeros((num_objs, 5), dtype=np.uint16)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            cls = CLASSES.index(obj.find('name').text.lower())

            gt_boxes[ix, :4] = [x1, y1, x2, y2]
            gt_boxes[ix, 4] = cls

        return gt_boxes


def loss_objects(im, scores, boxes, image_name, thresh, nms_tresh=0.3):
    """All detected objects on photo(with threshold)"""

#    print('\nimg name: ', image_name)
    gt_boxes = load_annotations(xml_name=image_name[:-3] + 'xml')

    curr_loss = [0, 0]

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]

        gt_boxes_class = gt_boxes[np.where(gt_boxes[:, 4] == cls_ind)]

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_tresh)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= thresh)[0]
	dets = dets[inds]
	
#	print('class = ', cls, '\ngt boxes\n', gt_boxes_class, '\ndets boxes\n', dets)

        for bbox in dets:
            for gt_bbox in gt_boxes_class:
		iou = bb_intersection_over_union(bbox, gt_bbox)
#		print(bbox, '\n\n', gt_bbox, '\n\t iou = ', iou)
#		print(iou)
                if iou > 0.4:
                    dets = np.delete(dets, np.where(dets == bbox), 0)
                    gt_boxes_class = np.delete(gt_boxes_class, np.where(gt_boxes_class == gt_bbox), 0)

        curr_loss[0] += len(dets)
        curr_loss[1] += len(gt_boxes_class)
#	print('class = ', cls, '\ndets\n', dets, '\n\ngts\n', gt_boxes_class)

    return curr_loss


def test(sess, net, image_name, thresh):
    """Detect object classes in an image using pre-computed object proposals."""

    im_file = os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'Images', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(sess, net, im)

    NMS_THRESH = 0.3

    return loss_objects(im, scores, boxes, image_name, thresh=thresh, nms_tresh=NMS_THRESH)


def testing(iter, end='test', thresh=0.9):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    demonet = 'vgg16'
    dataset = 'my_dataset'
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0].format(iter))
    print(tfmodel)


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    else:
        raise NotImplementedError
    net.create_architecture("TEST", len(CLASSES),
                           tag='default', anchor_scales=[4, 8, 16, 32], anchor_ratios=[0.15, 0.3, 0.5, 0.7])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    loss = np.array([0, 0])

    with open(os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'ImageSets', '{}.txt'.format(end)), 'r') as f:
        test_images = f.readlines()
        print(len(test_images))

    for ind, im_name in enumerate(test_images):
#	print(im_name[:-1] + '.jpg')
        loss += test(sess, net, im_name[:-1] + '.jpg', thresh=thresh)
        if ind % 20 == 0:
            print('curr loss: ', loss)
            print(len(test_images) - ind - 1, ' images left')

    return loss / len(test_images)


if __name__ == '__main__':
    if sys.argv[2] == 'all':
	threshs = [0.7 + 0.02 * i for i in range(12)]
    else:
    	threshs = float(sys.argv[2])
    
    with open(os.path.join(cfg.ROOT_DIR, 'real_results', 'real_results.txt'), 'w') as f_o:
	for thresh in threshs:
    	    error = testing(sys.argv[1], end='real', thresh=thresh)
    	    f_o.write(str(error[0]) + ' ' + str(error[1]) + str(thresh) + '\n')
