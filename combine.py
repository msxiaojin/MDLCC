import os, glob, re, signal, sys, argparse, threading, time, h5py, math, random
import scipy.misc 
import scipy.io
from skimage.measure import compare_ssim
from random import shuffle
from PIL import Image
from utils import *
# from nlrn_nobn import build_nlrn

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import multidomain_model as model


class combine(object):
    def __init__(self,Names, args):
        self.args = args
        self.camera_name = args.camera
        self.cv_index = args.cv_index
        
        self.Names = Names
        self.Num   = len(self.Names)
        self.imgNames = []
        self.cvSplits = {}
        idx = 0
        for name in self.Names:
            print('camera name -- %s\t '%( name  ))
            th_imgNames,th_cvSplits = get_dataset_info( name )
            self.imgNames = self.imgNames + th_imgNames
            cam_idx = np.zeros([len(th_imgNames),self.Num],dtype='float32')
            cam_idx[:,idx] = 1
                
            if idx == 0:
                self.cvSplits['train'] = th_cvSplits['train']
                self.cvSplits['valid'] = th_cvSplits['valid']
                self.cam_idx = cam_idx
            else:
                self.cvSplits['train'] = np.concatenate((self.cvSplits['train'],th_cvSplits['train']),axis = -1)
                self.cvSplits['valid'] = np.concatenate((self.cvSplits['valid'], th_cvSplits['valid']),axis = -1)
                self.cam_idx = np.concatenate((self.cam_idx,cam_idx),axis=0)
            idx += 1
            

        # training setting
        

    def test_per_dataset(self,camera_name, cv_fold): 
        # preparing test data
        test_batchsize = 2
        test_lists = []
        for i_dataset in camera_name:
            test_imgNames,test_cvSplits = get_dataset_info( i_dataset )
            cam_idx = np.zeros((len(test_imgNames),self.Num),'float32')
            idx     = [i for i in range(self.Num) if i_dataset==self.Names[i]]
            cam_idx[:,idx[0]] = 1
                
            test_no = [i for i in range(len(test_imgNames)) if test_cvSplits['valid'][cv_fold,i] == 1]
            for i in test_no :
                test_lists.append([test_imgNames[i], test_imgNames[i][:-4]+'_gt.mat' ,cam_idx[i,:] ])


        valid_imdb,label_imdb,index_imdb = set_test_imdb(test_lists,self.Num, self.testsize)

        
        model_list = sorted(glob.glob(os.path.join(self.model_path,"*epoch_*")))
        model_list = [fn for fn in model_list if os.path.basename(fn).endswith("meta")]
        model_names = [int(i.split('epoch_')[1].split('.ckpt')[0]) for i in model_list]
        model_names.sort(reverse=True)

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if 'CCD' in camera_name:
                self.test_input  = tf.placeholder(tf.float32, shape=(None, 1359, 2041, 3))
            else:
                self.test_input  = tf.placeholder(tf.float32, shape=(None, 512, 768, 3))
            self.test_idx    = tf.placeholder(tf.float32, shape=(None, 1, 1, self.Num))
            self.prob        = tf.placeholder(tf.float32)
            self.test_output,net = build_adaptive_mtl(self.test_input,self.test_idx,self.prob,
                                        reuse=self.reuse,scope='fc4_cv%d'%(cv_fold))
            self.test_output = tf.nn.l2_normalize(self.test_output,axis=-1)

            self.saver = tf.train.Saver()  
            
            # print(len(img_list))
            angular_errs = []
            metrics = []
            best_mean_metric   = 100*np.ones((5,))
            best_median_metric = 100*np.ones((5,))
            best_mean_epoch    = 3000
            best_median_epoch  = 3000
            
            for i_model in  range(len(model_names)):
                
                model_ckpt = os.path.join(self.model_path, "fc4_epoch_%03d.ckpt" % (model_names[i_model]) )
                self.saver.restore(sess, model_ckpt)

                offset = 0
                estimates = np.zeros( ( len(test_lists), 3) )
                
                s = time.time()
                for i_batch in range( len(test_lists)//test_batchsize+1): 
                    end_off = np.min( [len(test_lists),offset+test_batchsize] )
                    test_batch = valid_imdb[offset:end_off,:,:,:]
                    idx_batch  = index_imdb[offset:end_off,:,:,:]
                    feed_dict = {self.test_input:test_batch,self.test_idx:idx_batch,self.prob:1.0}
                    output = sess.run(self.test_output,feed_dict=feed_dict)
                    estimates[offset:end_off,:] = output
                    offset = end_off
                    del test_batch
                t = time.time()
                
                
                th_errors =  cal_angular_error_batch( estimates, label_imdb ) 
                angular_errs.append( th_errors ) 
                th_metrics = comp_metrics( th_errors )
                metrics.append(th_metrics)
               
                print( 'mean %.3f    median %.3f   tri %.3f   best25 %.3f   worst25 %.3f   time %.2f   epoch %d'%(th_metrics[0],
                                th_metrics[1],th_metrics[2],th_metrics[3],th_metrics[4],(t-s),model_names[i_model]))

                if th_metrics[0] <= np.array(metrics).min(axis=0)[0]:
                    best_mean_epoch = model_names[i_model]
                if th_metrics[1] <= np.array(metrics).min(axis=0)[1]:
                    best_median_epoch = model_names[i_model]
        return angular_errs,best_mean_epoch,best_median_epoch

def parse_args():
    parser = argparse.ArgumentParser(description='AWB arguments')
    parser.add_argument('--camera',type=str,default = 'all')
    parser.add_argument('--cv',dest='cv_index',type = int,default = 0)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set params
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # the order of cameras matters, decide the index of device-specific module
    # camera_names = ['NUS_ChengCanon600D','CCD'] 
    camera_names = ['NUS_ChengCanon1DsMkIII','Cube','CCD']
    # camera_names = ['Cube','NUS_ChengFujifilmXM1','NUS_ChengNikonD5200']
    camera_names = ['NUS_ChengCanon1DsMkIII','NUS_ChengCanon600D','NUS_ChengFujifilmXM1','NUS_ChengNikonD5200',
                    'NUS_ChengOlympusEPL6','NUS_ChengPanasonicGX1','NUS_ChengSamsungNX2000','NUS_ChengSonyA57',
                    'CCD','Cube']

    test_cam_name = ['NUS_ChengSonyA57']
    awb = combine(camera_names,args)
    awb.testsize = [1359,2041] #1234 1107
   
    errs = []
    mean_epoch = []
    median_epoch = []
    awb.reuse = False
    for i_cv_fold in range(3):
        # set model path   
        # save_name = 'nus2_ccd_mdlcc'+'_cv_%d'%(i_cv_fold)
        # save_name = 'nus1_cube_ccd_mdlcc'+'_cv_%d'%(i_cv_fold)
        # save_name = 'cube_nus3_nus4_mdlcc'+'_cv_%d'%(i_cv_fold)
        save_name = 'nus_ccd_cube_mdlcc'+'_cv_%d'%(i_cv_fold)
        

        
        awb.model_path = os.path.join( './/models',save_name )  
        
        th_errs,th_mean_epoch, th_median_epoch =  awb.test_per_dataset( test_cam_name,i_cv_fold )
        errs.append(th_errs)
        mean_epoch.append(th_mean_epoch)
        median_epoch.append(th_median_epoch)
        awb.reuse = True
    
    for i_cv in range(3):
        print( 'epoch for camera %s cv fold %d: %d and %d'%(test_cam_name[0],i_cv, mean_epoch[i_cv],median_epoch[i_cv]))
    
    best_mean   = np.ones((5,))*10
    best_median = np.ones((5,))*10
    for i in range(len(errs[0])):
        for j in range(len(errs[1])):
            for k in range(len(errs[2])):
                dataset_errs = errs[0][i] + errs[1][j] + errs[2][k]
                metrics = comp_metrics( dataset_errs )
                if metrics[0] < best_mean[0]:
                    best_mean = metrics
                if metrics[1] < best_median[1]:
                    best_median = metrics
    print(test_cam_name)
    print( "best median metric: mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t "%(
                                best_median[0],best_median[1],best_median[2],
                                best_median[3],best_median[4]) )
    print( "best mean metric: mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t "%(
                                best_mean[0],best_mean[1],best_mean[2],
                                best_mean[3],best_mean[4]) )
