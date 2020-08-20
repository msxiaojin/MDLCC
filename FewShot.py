import os, glob, re, signal, sys, argparse, threading, time, h5py, math, random
import scipy.misc 
import scipy.io

from random import shuffle
from utils import *

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class fewshot(object):
    def __init__(self,meta_names,Names,args):
        self.cropNum    = 4
        self.USE_QUEUE_LOADING = True
        self.num_thread = 2
        self.args = args
        self.cv_index = args.cv_index
        
        # training setting
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MAX_EPOCH  = args.MAX_EPOCH
        self.IMG_SIZE   = args.IMG_SIZE
        self.learning_rate = args.BASE_LR

        self.Names = Names
        self.Num   = len(meta_names)
        self.meta_imgNames = []
        self.fewshot_imgNames = []
        self.meta_names = meta_names
        self.cvSplits = {}
        idx = 0

        # set meta training dataset info.
        cvSplits_train = [[],[],[]]
        self.cam_idx = []
        for name in self.meta_names:
            print('camera name -- %s\t '%( name  ))
            th_imgNames,th_cvSplits = self.get_meta_dataset_info( name )
            self.meta_imgNames = self.meta_imgNames + th_imgNames
            cam_idx = np.zeros([len(th_imgNames),len(self.meta_names)],dtype='float32')
            cam_idx[:,idx] = 1
            self.cam_idx += list(cam_idx)
            idx += 1
            for fold in range(3):
                cvSplits_train[fold] +=  list(th_cvSplits['train'][fold,:])
          
        self.cvSplits['train'] = np.array(cvSplits_train)
        self.meta_cam_idx = np.array( self.cam_idx)

        # set few shot dataset info  
        print('camera name -- %s\t '%( self.Names  ))
        th_imgNames,th_cvSplits = self.get_meta_dataset_info(  self.Names )
        self.fewshot_imgNames += th_imgNames
        cam_idx = np.zeros([len(th_imgNames),len(self.meta_names)],dtype='float32')
        cam_idx[:,0] = 1
            
        self.fs_cvSplits = {}
        self.fs_cvSplits['train'] = th_cvSplits['train']
        self.fs_cvSplits['valid'] = th_cvSplits['valid']
        self.fs_cam_idx = cam_idx
            
            

        
    

    def fewshot_train(self,cv_fold,K):
        self.train_idx = self.fs_cvSplits['train'][cv_fold, :]
        train_no = [i for i in range(len(self.fewshot_imgNames)) if self.train_idx[i]==1]
        if K < len(train_no):
            np.random.seed( self.seed )
            np.random.shuffle(train_no)            
            train_no = train_no[:K]
        print('number of training pairs for few shot experiments',len(train_no))
        print( train_no)
        self.train_lists = []
        for i in train_no :
            self.train_lists.append([self.fewshot_imgNames[i], self.fewshot_imgNames[i][:-4]+'_gt.mat',self.fs_cam_idx[i,:]  ])
        print( 'cross validation fold: %d\t training numbers %d\t '%(cv_fold,len(self.train_lists)))
        shuffle( self.train_lists)
        
        
        
        print(self.model_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # set net architecture
        with tf.name_scope('all_fc4_cv%d'%(cv_fold)):
            global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        
            # set placeholder
            if self.USE_QUEUE_LOADING:
                print( "use queue loading"	)
                train_input_single  = tf.placeholder(tf.float32, shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3))
                train_gt_single  	= tf.placeholder(tf.float32, shape=(3,))
                train_idx_single    = tf.placeholder(tf.float32, shape=(self.Num,))
                
                
                q = tf.FIFOQueue(200, [tf.float32, tf.float32,tf.float32], 
                                                [[self.IMG_SIZE[0], self.IMG_SIZE[1], 3],
                                                [3],[self.Num]])
                enqueue_op = q.enqueue([train_input_single, train_gt_single,train_idx_single])           
                input_dequeue,gt_dequeue,idx_dequeue = q.dequeue_many(self.BATCH_SIZE*self.cropNum)
            
            # data augment 
            self.prob    = tf.placeholder(tf.float32)
            self.aug_ill = tf.placeholder(tf.float32, shape=(3,self.BATCH_SIZE*self.cropNum))
            self.train_input,self.train_gt,self.train_idx = data_augment(input_dequeue,gt_dequeue, 
                                    idx_dequeue, self.IMG_SIZE,self.BATCH_SIZE*self.cropNum,
                                    self.Num,  self.aug_ill)
             

            # build model, set trainable to False, except for camera-specific params
            train_output,net  = build_adaptive_mtl(self.train_input,self.train_idx,self.prob,
                                                reuse=self.reuse, scope='fc4_cv%d'%(cv_fold),trainable = False)
            
            self.train_output = tf.nn.l2_normalize(train_output,axis=-1)

            # loss function       
            loss = tf.reduce_sum( tf.multiply(self.train_output, self.train_gt), axis=-1) 
            loss = tf.clip_by_value( loss, -0.9999, 0.9999)
            loss =  180/math.pi *  tf.math.acos(loss) 
            self.camera_loss = tf.reduce_sum( loss)
            total_loss = self.camera_loss 

            # training vars
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                       scope=tf.get_variable_scope().name)

            all_train_vars = tf.trainable_variables() # denote the camera-specific params
            self.all_vars = all_vars[1:] # remove the first one, which is global_step
            for var in all_train_vars:
                print(var.name)
                total_loss += tf.nn.l2_loss(var)*1e-4

            # setting learning rate and optimizer
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=[1000,2000],
                                                            values=[0.0001,0.0001,0.0001])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name='cv_%d'%cv_fold ) 
            opt   = optimizer.minimize(total_loss,  global_step=global_step, var_list= all_train_vars,name='opt_%d'%cv_fold)

            # define saver
            self.saver = tf.train.Saver(self.all_vars,global_step, max_to_keep=0)
            
        # training
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            file_writer = tf.summary.FileWriter(".//logs//"+self.save_name, sess.graph)
            merged = tf.summary.merge_all()

            tf.initialize_all_variables().run()
            
            # loading previous training models
            start = 0
            model_lists = sorted(glob.glob(os.path.join(self.model_path,"*epoch_*.meta")))

            if model_lists:
                # if model_path not empty, continue training by loading previous model
                model_names = [int(i.split('epoch_')[1].split('.ckpt')[0]) for i in model_lists]
                model_names.sort()
                start = model_names[-1]
                print( "restore model from epoch %d\t"%(start))
                model_name = os.path.join(self.model_path, 'fc4_epoch_%03d.ckpt'%(start) )
                self.saver.restore(sess,model_name)
            else: # if empty model_path, then load meta model
                meta_lists = sorted(glob.glob(os.path.join(self.meta_path,"*epoch_*.meta")))
                meta_names = [int(i.split('epoch_')[1].split('.ckpt')[0]) for i in meta_lists]
                meta_names.sort()
                meta_ckpt  = os.path.join(self.meta_path,'fc4_epoch_%d.ckpt'%(meta_names[-1]))
                self.saver.restore(sess,meta_ckpt)
                
                # initialize the camera-specific weights, instead of inheriting from meta-model 
                all_train_vars[0] = tf.assign( all_train_vars[0],np.random.rand(1,1,7,24576) )
                all_train_vars[1] = tf.assign( all_train_vars[1],np.random.rand(1,1,7,4096) )
                

            ### WITH ASYNCHRONOUS DATA LOADING ###
            threads = []
            def signal_handler(signum,frame):
                sess.run(q.close(cancel_pending_enqueues=True))
                coord.request_stop()
                coord.join(threads)
                print( "Done")
                sys.exit(1)
            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal_handler)

            # without queue loading 
            if self.USE_QUEUE_LOADING:
                # create threads
                num_thread = self.num_thread
                coord = tf.train.Coordinator()
                for i in range(num_thread):
                    length = len(self.train_lists)//num_thread+1
                    t = threading.Thread(target=load_and_enqueue, args=(sess,coord, 
                                    self.train_lists[i*length:(i+1)*length],
                                    enqueue_op, train_input_single, 
                                    train_gt_single,train_idx_single,
                                    self.cropNum,self.IMG_SIZE, i)) 

                    threads.append(t)
                    t.start()
                print( "num thread:" , len(threads))
            
                
            train_loss  = np.zeros(self.MAX_EPOCH)
            
            smooth_loss = 0
            bs = self.BATCH_SIZE*self.cropNum
            for epoch in  range(start+1, self.MAX_EPOCH):
                error = 0
                cam_err = np.zeros(self.Num)
                count = np.zeros(self.Num)
                shuffle(self.train_lists)
                
                for step in range(max(1,len(self.train_lists)//self.BATCH_SIZE)):
                    
                    
                    aug_ill = np.random.uniform(low=0.8,high=1.2,size=(3,bs))
                    s = time.time()
                    if self.USE_QUEUE_LOADING:
                        feed_dict = {self.prob:0.5,self.aug_ill:aug_ill}
                    

                    
                    lr,_,th_cam_idx,l,output, g_step  = sess.run([learning_rate,opt, 
                                            self.train_idx,total_loss, train_output, global_step],
                                            feed_dict=feed_dict)
                    t = time.time()
                    # print( "[epoch %d: %d/%d] l2 loss %.3f\t lr %.6f\t time %.4f\t"%( epoch,step, 
                    #                                             len(self.train_lists)//(self.BATCH_SIZE),
                    #                                             np.sum(l)/self.BATCH_SIZE/self.cropNum,lr,(t-s)))
                    error = error + np.sum(l)/self.cropNum

                train_loss[epoch] = error / len(self.train_lists)

                print( "[epoch %d] total loss %.4f\t "%( epoch, train_loss[epoch]))

                
                if epoch % 50 ==0 or epoch == self.MAX_EPOCH-1:
                    self.saver.save(sess, self.model_path+"/fc4_epoch_%03d.ckpt" % epoch)
                    
        return 0

    def test_per_dataset(self,camera_name, cv_fold): 
        # preparing test data
        test_batchsize = 2
        test_lists = []
        for i_dataset in camera_name:
            test_imgNames,test_cvSplits = get_dataset_info( i_dataset )
            cam_idx = np.zeros((len(test_imgNames),self.Num),'float32')
            
            cam_idx[:,0] = 1
                
            test_no = [i for i in range(len(test_imgNames)) if test_cvSplits['valid'][cv_fold,i] == 1]
            for i in test_no :
                test_lists.append([test_imgNames[i], test_imgNames[i][:-4]+'_gt.mat' ,cam_idx[i,:] ])


        valid_imdb,label_imdb,index_imdb = set_test_imdb(test_lists,self.Num, self.testsize)

        # load few shot model
        model_list = sorted(glob.glob(os.path.join(self.model_path,"*epoch_*")))
        model_list = [fn for fn in model_list if os.path.basename(fn).endswith("meta")]
        model_names = [int(i.split('epoch_')[1].split('.ckpt')[0]) for i in model_list]
        model_names.sort(reverse=True)
        # model_names = model_names[:510]

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if 'CCD' in camera_name:
                self.test_input  = tf.placeholder(tf.float32, shape=(None, 1359, 2041, 3))
            else:
                self.test_input  = tf.placeholder(tf.float32, shape=(None, 512, 768, 3))
            self.test_idx    = tf.placeholder(tf.float32, shape=(None, 1, 1, self.Num))
            self.prob        = tf.placeholder(tf.float32)
            self.test_output,net = build_adaptive_mtl(self.test_input,self.test_idx,self.prob,
                                        reuse=self.reuse,scope='fc4_cv%d'%(cv_fold),trainable=False)
            # self.test_output = build_fc4_squeezenet(self.test_input,self.test_idx ,self.prob,
            #                         reuse=self.reuse,scope='fc4_cv%d'%(cv_fold),training=False)
            self.test_output = tf.nn.l2_normalize(self.test_output,axis=-1)

            if not hasattr(self, 'all_vars'):
                self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                scope=tf.get_variable_scope().name)
            self.saver = tf.train.Saver(self.all_vars) 
            
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
                    best_mean_metric = th_metrics
                    best_mean_errs   = th_errors
                if th_metrics[1] <= np.array(metrics).min(axis=0)[1]:
                    best_median_epoch = model_names[i_model]
                    best_median_metric = th_metrics
                    best_median_errs  = th_errors
        return best_mean_metric,best_median_metric,best_mean_errs,best_median_errs

    def awb(self, im, est ):
        im = im.squeeze()
        est = np.reshape( est,(1,1,3))
        im = im /est
        im = im / im.max()
        im = im **(1/2.2 )
        return im
        
    def get_meta_dataset_info(self,camera_name):
        if 'CCD' in camera_name:
            cameraFolder = './/database//CCD' 
            imgFolder = os.path.join( cameraFolder , 'preprocessed_512')
            imgNames  = sorted( glob.glob( os.path.join( imgFolder, '*.png')))
            cvName    = os.path.join( cameraFolder, 'CVsplit.mat')
            cvSplits  = get_cvsplit( cvName )
            
        elif 'NUS' in camera_name: 
            name = camera_name.split('NUS_')[-1]
            cameraFolder = os.path.join( './/database','NUS',name)
            imgNames  = sorted( glob.glob( os.path.join(cameraFolder,'preprocessed_512','*.png')))
            
            cvSplits  = get_nus_cvsplit(camera_name[4:])
        elif 'Cube' in camera_name :
            if camera_name == 'Cube_old':
                imgFolder = os.path.join( './/database//Cube_small' , 'preprocessed_512')
            else:
                imgFolder = os.path.join( './/database//Cube' , 'preprocessed_512')
            imgNames  = sorted( glob.glob( os.path.join( imgFolder, '*.png')))
            cvSplits = {}
            cvSplits['valid'] = np.zeros((3,len(imgNames)))
            
            cvSplits['valid'][0,0::3] = 1
            cvSplits['valid'][1,1::3] = 1
            cvSplits['valid'][2,2::3] = 1
            cvSplits['train'] = 1 - cvSplits['valid']
        
        imgNames = [i[:-4]+'.mat' for i in imgNames]
        return imgNames,cvSplits

def parse_args():
    parser = argparse.ArgumentParser(description='AWB arguments')
    # parser.add_argument('--camera',type=str,default = 'all')
    parser.add_argument('--cv',dest='cv_index',type = int,default = 2)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--k',dest='K',type=int,default=1) # number of few shot training samples
    parser.add_argument('--epoch',dest='MAX_EPOCH',type=int,default=1000)
    parser.add_argument('--tag',dest='tag',type=str,default='Cube_old',help='CCD or NUS1 or Cube_old') # test set
    parser.add_argument('--seed',dest='seed',type=int,default=0,help='random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # For few-shot evaluation, first run FewShot.py --cv 0, and FewShot.py --cv 1 
    # which will re-train the meta-model, on cross validation set 0&1
    # then run FewShot.py --cv 2, which will retrain the meta-model on cross validation set_2
    # and also perform validation on test dataset

    # set params
    args = parse_args()
    args.IMG_SIZE = (512, 512)
    args.BATCH_SIZE = 2
    args.BASE_LR = 0.0001
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # set up the meta camera names, the meta-model is trained using awb_demo.py with following camera_names
    meta_cam_names = ['NUS_ChengCanon600D','NUS_ChengFujifilmXM1','NUS_ChengNikonD5200',
                    'NUS_ChengOlympusEPL6','NUS_ChengPanasonicGX1','NUS_ChengSamsungNX2000',
                    'NUS_ChengSonyA57',]
    
    tag = args.tag
    if tag == 'CCD':
        test_cam_name  = 'CCD'
    elif tag == 'NUS1':
        test_cam_name = 'NUS_ChengCanon1DsMkIII'
    elif tag == 'Cube_old':
        test_cam_name = 'Cube_old'
    
    awb = fewshot(meta_cam_names,test_cam_name,args)
    awb.testsize = [1359,2041]
    awb.seed = args.seed 
    
    # load meta model
    K = args.K
    awb.reuse = False

    i_cv = args.cv_index
    awb.save_name = 'fewshot_'+tag +'_K%d_cv_%d'%(K,i_cv)
    awb.model_path = os.path.join('.//fewshot//','fewshot_'+tag +'_seed%d_K%d_cv_%d'%(awb.seed,K,i_cv))
    awb.meta_path  = os.path.join('.//fewshot','FW_meta_cv_%d'%(i_cv))
    
    # few-shot training model
    if K:
        awb.fewshot_train(i_cv,K)
        awb.reuse = True

    
    # if i_cv ==2, perform few-shot evaluation
    errs = []
    if i_cv == 2:
        for cv_fold in range(3):
            if K == 0:
                awb.model_path = os.path.join('.//fewshot','FW_' +tag +'_meta_cv_%d'%(cv_fold))
            else:
                awb.model_path = os.path.join('.//fewshot//','fewshot_'+tag +'_seed%d_K%d_cv_%d'%(awb.seed,K,cv_fold))
            _,_,th_best_mean, th_best_median =  awb.test_per_dataset( [test_cam_name],cv_fold )
            errs.append([th_best_mean, th_best_median])
            awb.reuse = True

        
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
