import tensorflow as tf
import os, glob, h5py, scipy, random, cv2, time, math, shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from random import shuffle
from rotate_and_crop import rotate_and_crop



def visawb( im, est ):
        im = im.squeeze()
        est = np.reshape( est,(1,1,3))
        im = im /est
        im = im / im.max()
        im = im **(1/2.2 )
        return im

def make_dir(folder):
    if not os.path.exists(folder):
            os.mkdir(folder)
    return 0

def copy_to_folder( ckpt_name, src,dest):
    files = glob.glob(os.path.join(src,'*'+ckpt_name+'*'))
    for i_file in files:
        shutil.copy2(i_file, dest)
    shutil.copy2(os.path.join(src,'checkpoint'), dest)
    return 0

def build_adaptive_mtl(x,camIdx,dropout, reuse,scope,trainable=True):
        
        net = []

        ###  ---------- common feature extraction stage. --------
        # Note we use the backbone (SqueezeNet) as in FC4
        x = tf.layers.conv2d(x,64,3,padding='valid',use_bias = True,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(3**2)/64)),
                    bias_initializer=tf.zeros_initializer(),
                    strides=[2,2],
                    activation=None,trainable=trainable,
                    name='conv01_stride2',
                    reuse=reuse)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool( x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')   

        x = fire_module(x,'fire02',16,64,64,reuse,trainable)
        x = fire_module(x,'fire03',16,64,64,reuse,trainable)
        x = tf.nn.max_pool( x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')   

        x = fire_module(x,'fire04',32,128,128,reuse,trainable)
        x = fire_module(x,'fire05',32,128,128,reuse,trainable)
        x = tf.nn.max_pool( x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')   

        x = fire_module(x,'fire06',48,192,192,reuse,trainable)
        x = fire_module(x,'fire07',48,192,192,reuse,trainable)
        x = fire_module(x,'fire08',48,192,192,reuse,trainable)
        global_feat = tf.reduce_mean( x, axis=[1,2],keepdims=True)
        x = tf.nn.max_pool( x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')   
        
        x = conv_layer(x, 64,6,True,'fc01',reuse,paddingMode='same',trainable=trainable)
        net.append(x)

        ### --------- adaptive camera-independent channel attention module -------------
        # first derive the fc weights: fc01 by convolution the cameraIndex, then applying batchwise conv
        # to derive the reweights (gamma) 
        fc01 = conv_layer( camIdx,384*64,1,False,name='cam_fc_01',reuse=reuse,paddingMode='valid')
        fc01 = tf.reshape(fc01, (-1,1,1,384,64))
        global_feat = batchwise_conv(global_feat,fc01,1,1,in_ch=384,out_ch=64)
        global_feat = tf.nn.leaky_relu( global_feat ,0.1)

        fc02 = conv_layer( camIdx,64*64,1,False,name='cam_fc_02',reuse=reuse,paddingMode='valid')
        fc02 = tf.reshape(fc02, (-1,1,1,64,64))
        global_feat = batchwise_conv(global_feat,fc02,1,1,in_ch=64,out_ch=64)
        gamma = tf.nn.sigmoid( global_feat)
        # applying channel reweighting gamma to achieve feature space transformation
        x = x * gamma 
        x = tf.nn.relu(x)
        net.append(x)

        ### --------final common illuminant estimation module ----------------
        x = tf.nn.dropout(x,dropout)
        x = conv_layer(x,3,1,True,'fc02',reuse,paddingMode='same',trainable=trainable)
        x = tf.reduce_sum( x, axis=[1,2])

        return x,net


def fire_module(x,name,s1x1,e1x1,e1x3,reuse,trainable=True):
    # squeeze layer, 1x1 conv layer
    x = conv_layer( x,s1x1,1,True,name+'_squeeze_1x1',reuse,'valid',trainable)
    x = tf.nn.relu(x)
    # x = tf.nn.leaky_relu(x,0.1)

    # expand layer
    e1 = conv_layer( x,e1x1,1,True,name+'_expand_1x1',reuse,'same',trainable)
    e3 = conv_layer( x,e1x3,3,True,name+'_expand_3x3',reuse,'same',trainable)
    x = tf.nn.relu( tf.concat((e1, e3),axis = -1))
    # x = tf.nn.leaky_relu(tf.concat((e1, e3),axis = -1),0.1)
    return x

def conv_layer(input,ch,filter_size,use_bias,name,reuse,paddingMode='same',trainable=True,stride=[1,1]):
    x = tf.layers.conv2d(input,ch,filter_size,padding=paddingMode,use_bias = use_bias,strides = stride,
                kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                bias_initializer=tf.zeros_initializer(),
                activation=None,
                name=name,trainable=trainable,
                reuse=reuse)
    return x


def batchwise_conv(x,F,fh,fw,in_ch,out_ch,s=1):
    # input x of shape NHWC
    # filter of size N*k*k*C*C
    inp = x
    # F = tf.reshape(F,[-1,fh,fw,channels,channels])
    F = tf.transpose(F, [1, 2, 0, 3, 4])
    F = tf.reshape(F, [fh, fw, -1, out_ch])

    inp_r = tf.transpose(inp, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
    # inp_r = tf.reshape(inp_r, [1, x.shape[1], x.shape[2], -1])
    inp_r = tf.reshape(inp_r, [1, tf.shape(x)[1], tf.shape(x)[2], -1])

    padding = "SAME" #or "SAME"
    out = tf.nn.depthwise_conv2d(
            inp_r,
            filter=F,
            strides=[1, s, s, 1],
            padding=padding) 

    
    out = tf.reshape(out, [tf.shape(out)[1], tf.shape(out)[2], -1, in_ch, out_ch])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_mean(out, axis=3)
    return out




def channel_attention(X,gamma):
    return gamma * (X)



def data_augment(train_batch,gt_batch, idx_batch, IMG_SIZE,bs, Num,aug_ill):
    # illuminant augment, up-down flip
    
    for i_batch in range(bs):
        
            th_crop = train_batch[i_batch,:,:,:]
            th_crop = tf.image.random_flip_left_right( th_crop )
            th_crop = tf.image.random_flip_up_down( th_crop )
            if i_batch%2 == 0:
                th_crop = tf.transpose(th_crop,(1,0,2))
            # obtaining gt
            th_gt  = gt_batch[i_batch,:]
            th_idx = tf.reshape(idx_batch[i_batch,:],(1,1,1,Num))
            # illuminant augment by relighting image and gt illuminant, accordingly
            th_crop   = th_crop * tf.reshape(aug_ill[:,i_batch],(1,1,3))
            th_gt     = np.multiply( th_gt , aug_ill[:,i_batch] )
            th_gt     = tf.nn.l2_normalize(th_gt,axis=-1)

            if i_batch  == 0:
                train_input = tf.expand_dims(th_crop,axis=0)
                train_gt    = tf.expand_dims( th_gt, axis=0)
                train_idx   = th_idx
            else:
                train_input = tf.concat((train_input,tf.expand_dims(th_crop,axis=0)),axis=0)
                train_gt    = tf.concat((train_gt,   tf.expand_dims( th_gt, axis=0)),axis=0)
                train_idx   = tf.concat((train_idx,  th_idx),axis=0)
            
    return tf.stop_gradient(train_input),tf.stop_gradient(train_gt),tf.stop_gradient(train_idx) 



def get_dataset_info(camera_name):
    if 'CCD' in camera_name:
        cameraFolder = './/database//CCD' 
        imgFolder = os.path.join( cameraFolder , 'full_preprocessed')
        imgNames  = sorted( glob.glob( os.path.join( imgFolder, '*.png')))
        cvName    = os.path.join( cameraFolder, 'CVsplit.mat')
        cvSplits  = get_cvsplit( cvName )

    elif 'NUS' in camera_name: 
        name = camera_name.split('NUS_')[-1]
        cameraFolder = os.path.join( './/database','NUS',name)
        imgNames  = sorted( glob.glob( os.path.join(cameraFolder,'preprocessed_512','*.png')))
                
        cvSplits  = get_nus_cvsplit(camera_name[4:])
    elif 'Cube' in camera_name:
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

def get_nus_cvsplit(camera_name):
    # The 8 sets in NUS datasets contain images from the same scene. 
    # To ensure the same scene would not be in both training and testing
    # we manually split the training and testing set for according to scene content
    # and saved in mat format
    cvname = './/database//NUS//cvsplits_nus.mat'
    nus_names  = scipy.io.loadmat( cvname )['nus'][0]
    cam_names = ['ChengCanon1DsMkIII','ChengCanon600D',
                'ChengFujifilmXM1','ChengNikonD5200',
                'ChengOlympusEPL6','ChengPanasonicGX1',
                'ChengSamsungNX2000','ChengSonyA57']
    # obtain all the scene index of NUS 8-cam datasets
    scene_index = []
    for i_dataset in range(len(nus_names)):
        scene_index += list(nus_names[i_dataset])
    scene_index = np.array( scene_index )
    # find how many unique scenes
    files = np.unique( scene_index )

    cam_idx = [i for i in range(8) if camera_name == cam_names[i]]
    cam_lists = np.squeeze( nus_names[cam_idx[0]])
    cv_split = {}
    # setting the splits for this camera
    cv_split['valid'] = np.zeros((3,len(cam_lists)))
    for i_seed in range(3):
        th_seed = files[i_seed::3]
        for i_file in range( len(cam_lists) ):
            cv_split['valid'][i_seed,i_file] = 1 if cam_lists[i_file] in th_seed else 0
        
    cv_split['train'] = 1 - cv_split['valid']
    return cv_split

def get_cvsplit(cvName):
    if os.path.isfile( cvName):
        data = scipy.io.loadmat( cvName )
        cv_split = {}
        cv_split['train'] = data['train_idx']
        cv_split['valid'] = data['test_idx']
    return cv_split

def load_ccd_gt(label_name):
    data = scipy.io.loadmat( label_name )
    gt   = data['real_rgb'].astype('float32')
    gt   = gt / np.tile( np.sqrt( np.sum( gt**2, 1,keepdims=True) ),(1,3))
    return gt

def load_nus_gt(label_name):
    data = scipy.io.loadmat( label_name )
    gt   = data['groundtruth_illuminants'].astype('float32')
    gt   = gt / np.tile( np.sqrt( np.sum( gt**2, 1,keepdims=True) ),(1,3))
    return gt

def load_image_batch(  imdb, offset, BATCH_SIZE,IMG_SIZE, cropNum):
    
    train_batch = []
    label_batch = []
    gtmap_batch = []
    idx_batch   = []

    for i_file in range(BATCH_SIZE):
        t  = imdb[offset + i_file]
        im = t.img
        gt = t.illum
        scale = random.uniform(0.3,1)
        s1 = time.time()
        th_img  = cv2.resize(im, (round(im.shape[0]*scale),round(im.shape[1]*scale)))
        t1 = time.time()
        print('cv resize image time %.6f'%(t1-s1))

        for j_patch in range( cropNum):
            
            start_x = random.randrange(0, th_img.shape[0] - IMG_SIZE[0] + 1)
            start_y = random.randrange(0, th_img.shape[1] - IMG_SIZE[1] + 1)
            th_img  = th_img[start_x:start_x + IMG_SIZE[0], start_y:start_y + IMG_SIZE[1],:]
            
            if (j_patch % 2) == 0:
                th_img = np.transpose(th_img,(1,0,2))

            th_gt    = gt / np.sqrt( np.sum( gt **2,axis=0))
            th_idx   = t.idx 
            th_gtmap = t.gt_map
            
            # stack into batch
            train_batch.append(th_img)
            label_batch.append(th_gt)
            gtmap_batch.append(th_gtmap)
            idx_batch.append( th_idx)
    train_batch = np.array( train_batch)
    label_batch = np.array( label_batch).squeeze()
    gtmap_batch = np.array( gtmap_batch)
    idx_batch = np.array( idx_batch)
    return train_batch,label_batch,idx_batch,gtmap_batch

def  set_valid_metrics(cameras):
    validation_metrics = {}
    for camera in cameras:
        validation_metrics[camera] = {}
        validation_metrics[camera]['error_over_epochs'] = [] 
        validation_metrics[camera]['best_mean_metric'] = [100,100,100,100,100]
        validation_metrics[camera]['best_median_metric'] = [100,100,100,100,100]

    return validation_metrics


def comp_metrics( errors ):
    errors  = np.squeeze( errors )
    
    percentiles = np.percentile(errors, [25,50,75,95] )
    mean   = np.mean(errors)
    mean2  = np.sqrt(np.mean(errors**2))
    mean4 = np.mean(errors**4)**(1/4)
    median = np.median( errors )
    tri = np.dot(percentiles[:3],[1, 2, 1])/4
    b25 = np.mean(errors[errors<=  percentiles[0]])
    w25 = np.mean(errors[errors>=  percentiles[2]])
    w05 = np.mean(errors[errors>=  percentiles[3]])
    mmax = np.max(errors)

    metrics = np.array( [mean,median,tri,b25,w25])
    return metrics

def cal_angular_error_batch(estimation, gt ):
    estimation  = np.reshape( estimation, (-1,3))
    gt          = np.reshape( gt, (-1,3))
    temp = np.sum(np.multiply(estimation , gt),axis=-1)/ np.linalg.norm(estimation,
                            axis=-1) / np.linalg.norm(gt,axis=-1)
    safe_v = 0.999999
    temp = np.clip( temp, -safe_v, safe_v)
    result = [math.acos(i)*180/math.pi for i in temp]
    return result




def load_and_enqueue(sess,coord, file_list, enqueue_op, train_input_single, 
                    train_gt_single, train_idx_single,
                    cropnum,IMG_SIZE,idx=0):
    count = 0
    length = len(file_list)
    try:
        while not coord.should_stop():
            i = count % length
            if i  == 0:
                shuffle(file_list)
            
            input_img = scipy.io.loadmat(file_list[i][0])['im']

            if input_img.shape[0] > input_img.shape[1]:
                input_img = np.transpose( input_img,(1,0,2))
            gt_img    = scipy.io.loadmat( file_list[i][1])['gt'].reshape([3])
            th_idx    = file_list[i][2]

            for i_crop in range( cropnum ):

                scale = random.uniform(0.1,1)
                s  = int(round(np.min(input_img.shape[0] * scale )))
                start_x = random.randrange(0, input_img.shape[0] - s + 1)
                start_y = random.randrange(0, input_img.shape[1] - s + 1)
                th_img  = input_img[start_x:start_x + s, start_y:start_y + s,:]
                # rotate image, copy from fc4 source code
                angle   = (random.random() - 0.5) * 60
                th_img  = rotate_and_crop(th_img, angle)

                th_img  = cv2.resize(th_img, (IMG_SIZE[0], IMG_SIZE[1]))
                sess.run(enqueue_op, feed_dict={train_input_single:th_img,
                                                train_gt_single:gt_img,
                                                train_idx_single:th_idx,
                                                })
            count+=1
    except Exception as e:
        print( "stopping...", idx, e)



def set_test_imdb(test_lists,Num,sze ):
    valid_imdb = []
    label_imdb = []
    index_imdb = []
    hist_imdb  = []
    for i_file in range(len(test_lists)):
        # print('processing test image...' ,i_file)
        valid_imname = test_lists[i_file][0]
        valid_im  = scipy.io.loadmat(valid_imname)['im']
        
        if valid_im.shape[0] > valid_im.shape[1]:
            valid_im = np.transpose( valid_im, (1,0,2))
        valid_im = valid_im[:sze[0],:sze[1],:]
        label    = scipy.io.loadmat( test_lists[i_file][1])['gt'].reshape((3))
        index    = np.reshape(test_lists[i_file][2],(1,1,-1))
        valid_imdb.append(valid_im)
        label_imdb.append(label)
        index_imdb.append(index)

    index_imdb = np.reshape( np.array(index_imdb),(-1,1,1,Num))
    return np.array( valid_imdb ).astype('float32'),np.array( label_imdb).astype('float32'),index_imdb.astype('float32') # np.array( hist_imdb ),


