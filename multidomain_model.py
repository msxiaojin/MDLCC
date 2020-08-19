import os, glob, re, signal, sys, argparse, threading, time, h5py, math, random
import scipy.misc 
import scipy.io
# from skimage.measure import compare_ssim
from random import shuffle
# from PIL import Image
# from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import *




class awb(object):
    def __init__(self,Names, args):
        self.USE_QUEUE_LOADING = True
        self.num_thread = 8
        self.args = args
        # self.camera_name = args.camera
        self.cv_index = args.cv_index
        
        self.Names = Names
        self.Num   = len(self.Names)
        self.imgNames = []
        self.cvSplits = {}
        idx = 0
        for name in self.Names:
            print('camera name -- %s\t '%( name  ))
            # read the training set image names and training split index for each camera
            th_imgNames,th_cvSplits = get_dataset_info( name )
            self.imgNames = self.imgNames + th_imgNames
            # set the device index, i.e., [1,0,0],[0,1,0],[0,0,1] for each camera, respectively
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
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MAX_EPOCH  = args.MAX_EPOCH
        self.IMG_SIZE   = args.IMG_SIZE
        self.learning_rate = args.BASE_LR
        self.cropNum    = 4 # randomly crop 4 patches from each training image for training
       

    def test(self,cv_fold):
        
        best_mean_metrics = []
        best_mean_epoch   = []
        best_median_metrics = []
        best_median_epoch   = []
        
        names = [i for i in self.Names]
        
        for i_camera in names:
            th_best_mean,th_mean_epoch, th_best_median,th_median_epoch = self.test_per_dataset([i_camera], cv_fold)

        best_mean_metrics.append(th_best_mean)
        best_mean_epoch.append(th_mean_epoch)
        best_median_metrics.append(th_best_median)
        best_median_epoch.append( th_median_epoch)
            
        return 0  

        

    def test_per_dataset(self,camera_name, cv_fold): 
        # preparing test data
        test_batchsize = 2
        self.save_model_path = os.path.join( './/models', self.save_name )  
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
        
            self.test_input  = tf.placeholder(tf.float32, shape=(None, None, None, 3))
            self.test_idx    = tf.placeholder(tf.float32, shape=(None, 1, 1, self.Num))
            self.prob        = tf.placeholder(tf.float32)
            self.test_output,net = build_adaptive_mtl(self.test_input,self.test_idx,self.prob,
                                        reuse=self.reuse,scope='fc4_cv%d'%(cv_fold))
            self.test_output = tf.nn.l2_normalize(self.test_output,axis=-1)

            if not hasattr(self, 'all_vars'):
                self.saver = tf.train.Saver()  
            else:
                self.saver = tf.train.Saver(self.all_vars)  
            
            # print(len(img_list))
            angular_errs = []
            metrics = np.zeros([len(model_list),5])
            best_mean_metric   = 100*np.ones((5,))
            best_median_metric = 100*np.ones((5,))
            best_mean_epoch    = 0
            best_median_epoch  = 0
            
            for i_model in  range(len(model_names)):
                epoch = model_names[i_model]
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
                metrics[epoch,:] = th_metrics
                
                 
                print( "Testing metric for camera: %s cv fold: %d epoch: %d; time: %.3f"%(camera_name,cv_fold,epoch,(t-s)))
                print( 'mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t'%(th_metrics[0],
                                th_metrics[1],th_metrics[2],th_metrics[3],th_metrics[4]))
                

                if th_metrics[0] < best_mean_metric[0]:
                    best_mean_metric = th_metrics
                    best_mean_epoch  = epoch
                if th_metrics[1] < best_median_metric[1]:
                    best_median_metric = th_metrics
                    best_median_epoch  = epoch
                print( "best median metric: mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t epoch %d"%(
                                best_median_metric[0],best_median_metric[1],best_median_metric[2],
                                best_median_metric[3],best_median_metric[4],best_median_epoch) )
                print( "best mean metric: mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t epoch %d"%(
                                best_mean_metric[0],best_mean_metric[1],best_mean_metric[2],
                                best_mean_metric[3],best_mean_metric[4],best_mean_epoch) )
            # move the best mean/median models to folder
            copy_to_folder( "fc4_epoch_%03d.ckpt" % (best_mean_epoch) ,self.model_path, self.save_model_path)
            copy_to_folder( "fc4_epoch_%03d.ckpt" % (best_median_epoch) ,self.model_path, self.save_model_path)

        
            print( self.model_path )        
            
        return best_mean_metric,best_mean_epoch,best_median_metric,best_median_epoch


    def train(self, cv_fold):
        # setting training and testing path
        self.train_idx = self.cvSplits['train'][cv_fold, :]
        train_no = [i for i in range(len(self.imgNames)) if self.train_idx[i]==1]
        self.train_lists = []
        # each training sample is represented as: imgname, gtname, and device index
        for i in train_no :
            self.train_lists.append([self.imgNames[i], self.imgNames[i][:-4]+'_gt.mat',self.cam_idx[i,:]  ])
        print( 'cross validation fold: %d\t training numbers %d\t '%(cv_fold,len(self.train_lists)))
        shuffle( self.train_lists)
        
        
        print(self.model_path)
        make_dir(self.model_path)
        # first save all training models in a 'temp' folder
        # then evaluate on test data to find the .ckpt which presents best metrics
        self.model_path = os.path.join(self.model_path,'temp')
        make_dir(self.model_path)



        # set net architecture
        with tf.name_scope('all_fc4_cv%d'%(cv_fold)):
            global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        
            # set placeholder
            if self.USE_QUEUE_LOADING:
                print( "----------------- use queue loading --------------"	)
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
             
            # build model for multitask training
            train_output,net  = build_adaptive_mtl(self.train_input,self.train_idx,self.prob,
                                                reuse=False, scope='fc4_cv%d'%(cv_fold),trainable = True)
            self.train_output = tf.nn.l2_normalize(train_output,axis=-1)

            # loss function       
            loss = tf.reduce_sum( tf.multiply(self.train_output, self.train_gt), axis=-1) 
            loss = tf.clip_by_value( loss, -0.9999, 0.9999)
            loss = 180/math.pi *  tf.math.acos(loss) 
            
            # if show loss on each camera
            temp =  tf.reshape(self.train_idx,(-1,self.Num)) * tf.reshape(loss,(-1,1) )
            self.camera_loss = tf.reduce_sum( temp, axis = 0)
            tf.summary.scalar('total_loss',tf.reduce_sum(loss,axis=0 )/self.BATCH_SIZE/self.cropNum)
            total_loss = tf.reduce_sum( self.camera_loss * self.loss_weight ,axis=0)

            # training vars
            all_vars = tf.trainable_variables()
            self.all_vars = all_vars
            for var in all_vars:
                print(var.name)
                total_loss += tf.nn.l2_loss(var)*1e-4
        
            # setting learning rate and optimizer
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=[1000,2000],
                                                            values=[0.0001,0.0001,0.0001])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name='cv_%d'%cv_fold ) 
            opt   = optimizer.minimize(total_loss,  global_step=global_step, var_list= self.all_vars)
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
                model_names = [int(i.split('epoch_')[1].split('.ckpt')[0]) for i in model_lists]
                model_names.sort()
                start = model_names[-1]
                print( "restore model from epoch %d\t"%(start))
                model_name = os.path.join(self.model_path, 'fc4_epoch_%03d.ckpt'%(start) )
                self.saver.restore(sess,model_name)


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
            camera_loss = np.zeros((self.MAX_EPOCH,self.Num))
            smooth_loss = 0
            bs = self.BATCH_SIZE*self.cropNum
            self.validation_metrics = set_valid_metrics(self.Names)
            for epoch in  range(start, self.MAX_EPOCH):
                error = 0
                cam_err = np.zeros(self.Num)
                count = np.zeros(self.Num)
                shuffle(self.train_lists)
                
                for step in range(len(self.train_lists)//self.BATCH_SIZE):
                    
                    
                    aug_ill = np.random.uniform(low=0.8,high=1.2,size=(3,bs))
                    s = time.time()
                    if self.USE_QUEUE_LOADING:
                        feed_dict = {self.prob:0.5,self.aug_ill:aug_ill}
                    else:
                        offset = step*self.BATCH_SIZE
                        s2 = time.time()
                        th_inputs,th_gts,th_idx,th_gtmaps = load_image_batch(  imdb, offset, self.BATCH_SIZE,self.IMG_SIZE, self.cropNum)
                        t2 = time.time()
                        print('load image batch time %.6f'%(t2-s2))
                        feed_dict = {input_dequeue:th_inputs,gt_dequeue:th_gts,idx_dequeue:th_idx,
                                    gtmap_dequeue:th_gtmaps,self.prob:0.5,self.aug_ill:aug_ill}

                    lr,train_summary,_,camera_error,th_cam_idx,l,output, g_step  = sess.run([learning_rate,merged,opt, 
                                            self.camera_loss,self.train_idx,total_loss, train_output, global_step],
                                            feed_dict=feed_dict)
                    t = time.time()
                    print( "[epoch %d: %d/%d] l2 loss %.3f\t lr %.6f\t time %.4f\t"%( epoch,step, 
                                                                len(self.train_lists)//(self.BATCH_SIZE),
                                                                np.sum(l)/self.BATCH_SIZE/self.cropNum,lr,(t-s)))
                    error = error + np.sum(l)/self.cropNum
                    cam_err = cam_err + camera_error/self.cropNum
                    count   = count + np.sum(th_cam_idx,axis=0)/self.cropNum
                    file_writer.add_summary(train_summary,g_step)
                train_loss[epoch] = error / len(self.train_lists)
                camera_loss[epoch] = cam_err / count
                
                print( "[epoch %d] total loss %.4f\t "%( epoch, train_loss[epoch]))
                print('loss for each camera:')
                print(camera_loss[epoch])
                
                self.save_this_epoch = True
                # self.validation(sess,cv_fold,epoch) # not efficient
                if self.save_this_epoch:
                    self.saver.save(sess, self.model_path+"/fc4_epoch_%03d.ckpt" % epoch)
        return 0

    def validation(self,sess,cv_fold,epoch):
        test_input  = tf.placeholder(tf.float32, shape=(None,None, None, 3))
        test_gt  	= tf.placeholder(tf.float32, shape=(None,3))
        test_idx    = tf.placeholder(tf.float32, shape=(None,1,1,self.Num))
        test_output,_ = build_adaptive_mtl(test_input,test_idx,self.prob,
                                    reuse=True,scope='fc4_cv%d'%(cv_fold))
        test_output = tf.nn.l2_normalize(test_output,axis=-1)

        sze = self.testsize
        for i_dataset in self.Names:
            test_imgNames,test_cvSplits = get_dataset_info( i_dataset )
            cam_idx = np.zeros((1,1,1,self.Num),'float32')
            idx     = [i for i in range(self.Num) if i_dataset==self.Names[i]]
            cam_idx[:,:,:,idx[0]] = 1
                
            test_no = [i for i in range(len(test_imgNames)) if test_cvSplits['valid'][cv_fold,i] == 1]
            
            
            for i_file in test_no:
                valid_imname = test_imgNames[i_file]
                valid_im  = scipy.io.loadmat(valid_imname)['im']
                
                if valid_im.shape[0] > valid_im.shape[1]:
                    valid_im = np.transpose( valid_im, (1,0,2))
                valid_im = valid_im[:sze[0],:sze[1],:]
                label    = scipy.io.loadmat( test_imgNames[i_file][:-4]+'_gt.mat' )['gt'].reshape((3))
                index    = cam_idx

                feed_dict = {test_input:valid_im[np.newaxis,:,:,:],test_idx:index,self.prob:1.0}
                output = sess.run(test_output,feed_dict=feed_dict)
                errors.append( cal_angular_error_batch( output, label ) )

            
                
            errors =  np.array( errors )
            th_metrics = comp_metrics( errors ) 
            self.validation_metrics[i_dataset]['error_over_epochs'].append(th_metrics)               
            
            print( "---------- Validation metric for camera: %s cv fold: %d epoch: %d---------------"\
                                                        %(i_dataset,cv_fold,epoch))
            print( 'mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t'%(th_metrics[0],
                                th_metrics[1],th_metrics[2],th_metrics[3],th_metrics[4]))

            if th_metrics[0] < self.validation_metrics[i_dataset]['best_mean_metric'][0]:
                self.validation_metrics[i_dataset]['best_mean_metric'] = th_metrics
                self.save_this_epoch = True
            if th_metrics[1] < self.validation_metrics[i_dataset]['best_median_metric'][1]:
                self.validation_metrics[i_dataset]['best_median_metric'] = th_metrics
                self.save_this_epoch = True

            best_mean_metric = self.validation_metrics[i_dataset]['best_mean_metric']
            best_median_metric = self.validation_metrics[i_dataset]['best_median_metric']
            print( "best median metric: mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t "%(
                            best_median_metric[0],best_median_metric[1],best_median_metric[2],
                            best_median_metric[3],best_median_metric[4]) )
            print( "best mean metric: mean %.3f\t median %.3f\t tri %.3f\t best25 %.3f\t worst25 %.3f\t "%(
                            best_mean_metric[0],best_mean_metric[1],best_mean_metric[2],
                            best_mean_metric[3],best_mean_metric[4]) )

        return 0
    
    
    
        





 