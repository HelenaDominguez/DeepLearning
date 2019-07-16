def train_convnet_GZOO(X,Y,ntrain,nval,test_name,weights_file_name):


    ind=random.sample(range(0, ntrain+nval-1), ntrain+nval-1)
    X_train = X[ind[0:ntrain],:,:,:]   
    X_val = X[ind[ntrain:ntrain+nval],:,:,:]
    Y_train = Y[ind[0:ntrain]]
    Y_val = Y[ind[ntrain:ntrain+nval]]
    
    
    ## Params
    # model params
    batch_size = 30 #64
    nb_epoch = 50
    data_augmentation = True
    normalize = False
    y_normalization = False
    norm_constant = 255 

    # SGD parameters
    lr=0.001   #0.001
    decay=0
    momentum=0.9   #0.9
    nesterov=True

    depth=32
    nb_dense = 64

    #output params
    verbose = 1
    print("Test name is: " + test_name)

    # input image dimensions
    img_rows, img_cols = X_train.shape[2:4]
    img_channels = 3
    print(img_rows, img_cols, img_channels)

    ### Right shape for X
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_val = X_val.reshape(X_val.shape[0], img_channels, img_rows, img_cols)


    #Avoid more iterations once convergence
    patience_par=10
    earlystopping = EarlyStopping( monitor='val_loss',patience = patience_par,verbose=0,mode='auto' )
    modelcheckpoint = ModelCheckpoint(test_name+"_best.hd5",monitor='val_loss',verbose=0,save_best_only=True)


    #========= Model definition (as in GZOO convnet)=======
    
    model = Sequential()
    model.add(Convolution2D(32, 6,6, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))

    #Neuronal Networks 
    #---------------------#
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5)) 
    model.add(Dense(1, init='uniform', activation='sigmoid'))
   
    print("Compilation...")
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    print("Model Summary")
    print("===================")
    model.summary()
    
    
    # Define non-trainable layers
    for layer in model.layers[:Nlayer]:     
        layer.trainable = False

    # Transfer learning: recover weights from previus model
    if os.path.isfile(weights_file_name) and RECOVER_MODEL:
        print("Loading model", weights_file_name)
        model.load_weights(weights_file_name)
        
        
   #Data Augmentation
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val),
                            shuffle=True,
                            verbose=verbose)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False, 
            samplewise_center=False, 
            featurewise_std_normalization=False, 
            samplewise_std_normalization=False,
            zca_whitening=False, 
            rotation_range=45,
            width_shift_range=0.05,  
            height_shift_range=0.05, 
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=[0.75,1.3])  

        
        datagen.fit(X_train)
        
        history = model.fit_generator(
                    datagen.flow(X_train, Y_train, batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(X_val, Y_val),
                    callbacks=[modelcheckpoint]
                )


    # save weights
    print("Saving model...")
    model.save_weights(test_name+".hd5",overwrite=True)
    
        
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    return


#############################################################

def validate_convnet_GZOO(X,model_name):
    
    ## Params
    # model params
    batch_size = 30 #64
    nb_epoch = 50
    data_augmentation = True
    normalize = False
    y_normalization = False
    norm_constant = 255 

    # SGD parameters
    lr=0.001   #0.001
    decay=0
    momentum=0.9   #0.9
    nesterov=True

    depth=32
    nb_dense = 64

    #output params
    verbose = 1

    # input image dimensions
    img_rows, img_cols = X.shape[2:4]
    img_channels = 3
    print ("This is where you are")
    print(img_rows, img_cols)

    print "X", X.shape
    print ("You are also here")
    ### Right shape for X
    X = X.reshape(X.shape[0], img_channels, img_rows, img_cols)
    
    

    #========= Model definition=======
    
    model = Sequential()
    model.add(Convolution2D(32, 6,6, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))

    #Neuronal Networks 
    #---------------------#

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5)) 
    model.add(Dense(1, init='uniform', activation='sigmoid'))
   
    print("Compilation...")
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
 

    
    #====== load model & predict=======

    print("model", model_name)
    model.load_weights(model_name+".hd5")

   
    for layer in model.layers:
        weights = layer.get_weights()    
    
    Y_pred = model.predict_proba(X)

    return Y_pred



    #############################################################
        
from astropy.io import fits
from astropy.table import Table
from math import log10
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Sersic2D
import os
from scipy.misc import imresize
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from scipy import misc
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
import glob
import theano
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.engine.topology import Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import rmsprop
import random
import pdb
from sklearn.utils import shuffle

from keras import backend as K
K.set_image_dim_ordering('th')
                         
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.models import model_from_json
from keras.layers import Dense
from keras.models import model_from_yaml



RECOVER_MODEL=True
TRAIN_MODEL=True
TEST_MODEL=True


maxim=4774L  #number of images read in D, Y vector
pathin="pathin"
nparams=1
    

data=fits.getdata(pathin+"DECALS_DES_S82_4938.fit",1)
ID_all=data['COADD_OBJECTS_ID']
Thumb=data['THUMBNAME']

#DECALS votes
Ps_all=data['t00_smooth_or_features_a0_smooth_weighted_frac']+1e-6
Pd_all=data['t00_smooth_or_features_a1_features_weighted_frac']+1e-6


## Loading image vector
print("Loading D, Y")
D_nosel=np.load(pathin+"image_vector_"+str(maxim)+".npy") #image matrix
ID2=np.load(pathin+"target_vector_"+str(maxim)+".npy")    #name imag
ID=np.load(pathin+"ID_vector_"+str(maxim)+".npy")         #ID catalogue


#look for galaxies with cutouts (only 4774)
ind=[]
for i in ID:
    kk=np.where(ID_all==i)
    print(i, kk)
    ind.append(kk[0][0])

Ps=Ps_all[ind] # P_smooth from GZ2
Pd=Pd_all[ind] # P_disk from GZ2


#Select well defined training sample
ok=np.where((Ps > 0.7) | (Pd > 0.7)) #2437
D=D_nosel[ok[0], :, :, :]

Y=np.zeros(ok[0].shape)
Y[np.where(Pd[ok] > 0.7)]=1 #854 disk vs 1583 smooth



##normalization of input matrix
mu = np.amax(D,axis=(2,3))
print mu.shape
for i in range(0,mu.shape[0]):
    #print i
    D[i,0,:,:] = D[i,0,:,:]/mu[i,0]
    D[i,1,:,:] = D[i,1,:,:]/mu[i,1]
    D[i,2,:,:] = D[i,2,:,:]/mu[i,2]


print D.shape, Y.shape



#========= Train model ===============

if TRAIN_MODEL:
    
    ntrainv=[500, 1000]    # loop in size of training sample
    nweight=[20000]

    for i in ntrainv:
        print(i, "training sample size")
        ntrain=i
        nval=i/10 #validation sample size

        model_name=pathin+"DES_model_t00_class_frac_wtZOO_"+str(ntrain)+"" #new model name

        # Transfer knowdelege from SDSS model
         path_model="/path_model"
         weights_file_name=path_model+"ZOO_model_t00_class_frac_wt_"+str(nweight[0])+".hd5"
        print(weights_file_name)
        

        print("Training Model")
        print("====================")
        train_convnet_GZOO(D,Y,ntrain,nval,model_name, weights_file_name)



#========= Test model ===============

 
if TEST_MODEL:

        #test with galaxies not  used in training sample
        pred_index=ok[0][2200]
        npred=np.shape(D_nosel)[0]-pred_index
        
        path_model="/path_model"

        for mod in glob.glob(path_model+"*.hd5"):

            tmp=mod.split("/")
            tmp=tmp[4].split(".")
            model_name=tmp[0]
            
            print(model_name)
            model=path_model+model_name


            print("Validating model")
            print("====================")
            Y_pred=validate_convnet_GZOO(D_nosel[pred_index:pred_index+npred, :, :, :],model) 

            pdb.set_trace() 
               
            print("Saving Y_pred.fit")
            print("====================")
            col1 = fits.Column(name='COADD_OBJECTS_ID', format='20A', array=ID[pred_index:pred_index+npred])
            col2 = fits.Column(name='a0_1_pred', format='F', array=Y_pred)

            cols = fits.ColDefs([col1,col2])
            tbhdu = fits.BinTableHDU.from_columns(cols)
            tbhdu.writeto(pathin+model_name+"_nosel.fit",clobber='True')

        

pdb.set_trace()     
