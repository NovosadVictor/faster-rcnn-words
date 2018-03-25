#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#my imports
from os import listdir
from os.path import isfile, join

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

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_30000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'my_dataset': ('my_dataset_train', ), 'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

RESULT_PATH = os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'results')

def all_objects(scores, boxes, image_name, thresh=0.8, nms_tresh=0.3):
    """All detected objects on photo(with threshold)"""
    end = os.path.splitext(image_name)[-1]
    print(image_name, end)
    with open(os.path.join(RESULT_PATH, image_name[:-len(end)] + '.txt'), 'w') as f:
        f.write(end + '\n')
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, nms_tresh)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= thresh)[0]

            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]

                list_to_file = [cls, score, bbox[0], bbox[1], bbox[2], bbox[3]]
                for ele in list_to_file:
                    f.write(str(ele) + ' ')
                f.write('\n')



def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
#    im_file = os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'fake_imgs', image_name)
#    im_file = os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'real_imgs', image_name)
    im_file = os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'real_imgs', image_name)
    print(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    # My code
    all_objects(scores, boxes, image_name, thresh=CONF_THRESH, nms_tresh=NMS_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [my_dataset]',
                        choices=DATASETS.keys(), default='my_dataset')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


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
#    elif demonet == 'res101':
#        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", len(CLASSES),
                          tag='default', anchor_scales=[4, 8, 16, 32], anchor_ratios=[0.15, 0.3, 0.5, 0.7])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

#    with open(os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'ImageSets', 'test.txt'), 'r') as f:
#        test_images = f.readlines()
#    test_images = [x.strip() for x in test_images]
#    test_images = test_images[0::5]
#    test_images.append('00000')
    test_images = [f for f in listdir(os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'real_imgs'))]
#    fake_images = [f for f in listdir(os.path.join(cfg.ROOT_DIR, 'text_img_dataset', 'data', 'fake_imgs'))]
    print(len(test_images))
#    print(len(fake_images))


    for ind, im_name in enumerate(test_images):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#        print('Demo for text_img_dataset/data/Images/{}'.format(im_name))
        print('Demo for text_img_dataset/data/real_imgs/{}'.format(im_name))
        demo(sess, net, im_name)
#        demo(sess, net, im_name + '.jpg')
        print(len(test_images) - ind - 1, ' images left')
#    for ind, im_name in enumerate(fake_images):
#        print('Demo for text_img_dataset/data/fake_imgs/{}'.format(im_name))
#        demo(sess, net, im_name)
#        print(len(fake_images) - ind - 1, ' images left')
