import os, glob, re, signal, sys, argparse, threading, time, h5py, math, random
# import scipy.misc 
# import scipy.io
# from skimage.measure import compare_ssim
# from random import shuffle
# from PIL import Image
from utils import *
# import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import multidomain_model as model


def parse_args():
    parser = argparse.ArgumentParser(description='AWB arguments')
    # parser.add_argument('--camera',type=str,default = 'all')
    parser.add_argument('--cv',help='the cross validation fold',dest='cv_index',type = int,default = 2)
    parser.add_argument('--gpu',dest='gpu_id', type=str, default='0')
    parser.add_argument('--epoch',dest='MAX_EPOCH',type=int,default=2000)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set params
    args = parse_args()

    args.IMG_SIZE = (512, 512)
    args.BATCH_SIZE = 2
    args.BASE_LR = 0.0001
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    i_cv_fold = args.cv_index

    ## the combined cameras in multi-domain learning 
    # camera_names = ['NUS_ChengCanon600D','CCD']
    # camera_names = ['NUS_ChengCanon1DsMkIII','Cube','CCD']
    # camera_names = ['Cube','NUS_ChengFujifilmXM1','NUS_ChengNikonD5200']
    camera_names = ['NUS_ChengCanon1DsMkIII','NUS_ChengCanon600D','NUS_ChengFujifilmXM1','NUS_ChengNikonD5200',
                    'NUS_ChengOlympusEPL6','NUS_ChengPanasonicGX1','NUS_ChengSamsungNX2000','NUS_ChengSonyA57',
                    'CCD','Cube']
    
    
    awb = model.awb(camera_names,args)
    awb.testsize = [1359,2041]
    awb.loss_weight = np.array( [1])
    awb.VISUALIZE = False
    awb.ccd_test_epoches = 1500
    awb.USE_QUEUE_LOADING = True
 
    awb.save_name = 'nus_ccd_cube_mdlcc'+'_cv_%d'%(i_cv_fold)
    awb.model_path = os.path.join( './/models', awb.save_name )  
    
    ## training for each cv fold
    awb.reuse = False
    awb.train( i_cv_fold )

    ## testing for each cv fold
    awb.reuse = True
    awb.test( i_cv_fold ) 
    


    