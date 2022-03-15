import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";


import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from utility import LoadDataNoDefCW	

n_closed = 200
n_max=3

def accuracy(Inliers_true, other_true, Inliers_score_raw,  other_score_raw, num_classes):

	Inliers_pred = np.argmin(Inliers_score_raw, axis=1)
	other_pred = np.argmin(other_score_raw, axis=1)

	Inliers_score = np.min(Inliers_score_raw, axis=1)
	other_score = np.min(other_score_raw, axis=1)
	
	end = np.min(Inliers_score)
	start = np.max(Inliers_score)
	gap = -0.002 #(end- start)/200000 # precision:200000
	best_acc_ave= 0.0
	best_threshold = start
	accuracy_thresh = 90.0 
	accuracy_range = np.arange(start, end, gap)
	for i, delta in enumerate(accuracy_range):
		# samples with prediction probabilities less than thresh are labeld as open
		Inliers_label = np.where(Inliers_score<delta, Inliers_pred, num_classes)
		# Calculate accuracy 
		a = np.sum(np.where(Inliers_true == Inliers_label, 1, 0))/Inliers_label.shape[0]*100       
		Outliers_label = np.where(other_score<delta, other_pred, num_classes)       
		b = np.sum(np.where(other_true == Outliers_label, 1, 0))/Outliers_label.shape[0]*100  
		# print('i:{}, delta:{}, close:{}, open:{}'.format(i, delta, a, b))

		if i==0 and a<accuracy_thresh:
			print('Closed set accuracy did not reach 90')
			return a, b, delta   

		# if (a+b)/2 >best_acc_ave and a>=90.:
		if a<accuracy_thresh and i>0:
			print('ideal')
			delta = accuracy_range[i-1]
			best_threshold = delta

			Inliers_label = np.where(Inliers_score<=delta, Inliers_pred, num_classes)
			# Calculate accuracy 
			best_acc_inlier = np.sum(np.where(Inliers_true == Inliers_label, 1, 0))/Inliers_label.shape[0]*100       
			Outliers_label = np.where(other_score<delta, other_pred, num_classes)       
			best_acc_outlier = np.sum(np.where(other_true == Outliers_label, 1, 0))/Outliers_label.shape[0]*100  
			
			best_acc_ave = (best_acc_inlier+best_acc_outlier)/2
			# print('\ti:{}, delta:{}, close:{}, open:{}'.format(i, delta, a, b))
			
			return best_acc_inlier, best_acc_outlier, best_threshold

		if i==len(accuracy_range)-1:
			print('Closed set accuracy did not fall below 90')
			return best_acc_inlier, best_acc_outlier, best_threshold
	

def auroc(inDataMin, outDataMin, title='test', trial_num=1):
	# print(inData.shape, outData.shape)
	# inDataMin = np.min(inData, 1)
	# outDataMin = np.min(outData, 1)

	Y1 = outDataMin
	X1 = inDataMin
	start = np.max([np.max(X1), np.max(Y1)])
	end = np.min([np.min(X1),np.min(Y1)])
	# end = np.min(X1)
	# start = np.max(X1)
	gap = (end- start)/200000

	print(start, end, gap)

	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 <= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
		# f1.write("{}\n".format(tpr))
		# f2.write("{}\n".format(fpr))
		aurocBase += (-fpr+fprTemp)*tpr
		# print('delta:{}, tpr:{}, fpr:{}, fprTemp:{}, aurocBase:{}'.format(delta, 	tpr, fpr, fprTemp, aurocBase))
		fprTemp = fpr

	return aurocBase

def get_f1(Inliers_true, other_true, Inliers_pred, Outliers_pred,  num_classes):
	 
	'''
	delta: threshold value to reject open samples calculated from method 'accuracy'
	'''
	# Inliers_label = []
	# for i in range(Inliers_score.shape[0]):
	# 	Inliers_label.append(np.where(Inliers_score[i]<=delta[int(Inliers_pred[i])], Inliers_pred[i], num_classes))

	# Inliers_label = np.array(Inliers_label)

	# Outliers_label = []
	# for i in range(other_score.shape[0]):
	# 	Outliers_label.append(np.where(other_score[i]<=delta[int(Outliers_pred[i])], Outliers_pred[i], num_classes))

	# Outliers_label = np.array(Outliers_label)
	# prediction = np.append(Inliers_label, Outliers_label, axis=0)

	prediction = np.append(Inliers_pred, Outliers_pred, axis=0)
	labels = np.append(Inliers_true, other_true, axis=0)

	# calculate confusion matrix
	mat = np.zeros((num_classes+1, num_classes+1))

	# y-axis: label, x-axis:prediction
	for i in range(prediction.shape[0]):
		mat[int(labels[i]), int(prediction[i])] = mat[int(labels[i]), int(prediction[i])] + 1

	P=0
	R=0
	for c in range(0, num_classes):
		tp = np.diagonal(mat)[c]
		fp = np.sum(mat[:, c])-tp
		fn = np.sum(mat[c, :])-tp

		P = P+(tp/(tp+fp))
		R = R+(tp/(tp+fn))



	P = P/num_classes
	R = R/num_classes

	F = 2*P*R/(P+R)

	return F

def get_small_set(x, y, n):
	ind = np.zeros((1,))
	for i in range(0, n_closed):
		inds = np.where(y==i)[0]
		ind = np.append(ind, inds[0:n], axis=0)

	ind = ind[1:].astype(int)
	return x[ind], y[ind]




def build(n_closed, filepath, layer):
	from keras.models import Model
	from keras.layers import Dense, Layer, Input, Flatten
	from keras.layers import Dropout, Activation, BatchNormalization
	from keras.layers import Conv1D, MaxPooling1D 
	from keras.layers.advanced_activations import ELU
	from keras.optimizers import Adamax

	filter_num = ['None',32,64,128,256]
	kernel_size = ['None',8,8,8,8]
	conv_stride_size = ['None',1,1,1,1]
	pool_stride_size = ['None',4,4,4,4]
	pool_size = ['None',8,8,8,8]

	length = 1500
	input_shape = (length, 1)
	
	inp = Input(shape=(length, 1))
	x1 = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
						 strides=conv_stride_size[1], padding='same',
						 name='block1_conv1')(inp)
	x2 = BatchNormalization(axis=-1)(x1)
	x3 = ELU(alpha=1.0, name='block1_adv_act1')(x2)
	x4 = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
						 strides=conv_stride_size[1], padding='same',
						 name='block1_conv2')(x3)
	x5 = BatchNormalization(axis=-1)(x4)
	x6 = ELU(alpha=1.0, name='block1_adv_act2')(x5)
	x7 = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
							   padding='same', name='block1_pool')(x6)
	x8 = Dropout(0.1, name='block1_dropout')(x7)

	x9 = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
						 strides=conv_stride_size[2], padding='same',
						 name='block2_conv1')(x8)
	x10 = BatchNormalization()(x9)
	x11 = Activation('relu', name='block2_act1')(x10)

	x12 = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
						 strides=conv_stride_size[2], padding='same',
						 name='block2_conv2')(x11)
	x13 = BatchNormalization()(x12)
	x14 = Activation('relu', name='block2_act2')(x13)
	x15 = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
							   padding='same', name='block2_pool')(x14)
	x16 = Dropout(0.1, name='block2_dropout')(x15)

	x17 = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
					 strides=conv_stride_size[3], padding='same',
					 name='block3_conv1')(x16)
	x18 = BatchNormalization()(x17)
	x19 = Activation('relu', name='block3_act1')(x18)
	x20 = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
					 strides=conv_stride_size[3], padding='same',
					 name='block3_conv2')(x19)
	x21 = BatchNormalization()(x20)
	x22 = Activation('relu', name='block3_act2')(x21)
	x23 = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
							   padding='same', name='block3_pool')(x22)
	x24 = Dropout(0.1, name='block3_dropout')(x23)

	x25 = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
					 strides=conv_stride_size[4], padding='same',
					 name='block4_conv1')(x24)
	x26 = BatchNormalization()(x25)
	x27 = Activation('relu', name='block4_act1')(x26)
	x28 = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
					 strides=conv_stride_size[4], padding='same',
					 name='block4_conv2')(x27)
	x29 = BatchNormalization()(x28)
	x30 = Activation('relu', name='block4_act2')(x29)
	x31 = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
						   padding='same', name='block4_pool')(x30)
	x32 = Dropout(0.1, name='block4_dropout')(x31)

	x33 = Flatten(name='flatten')(x32)
	x34 = Dense(512,  name='fc1')(x33)
	x35 = BatchNormalization()(x34)
	x36 = Activation('relu', name='fc1_act')(x35)

	x37 = Dropout(0.7, name='fc1_dropout')(x36)

	x38 = Dense(512, name='fc2')(x37)
	x39 = BatchNormalization()(x38)
	x40 = Activation('relu', name='fc2_act')(x39)

	x41 = Dropout(0.5, name='fc2_dropout')(x40)

	x42 = Dense(n_closed, name='fc3')(x41)
	x43 = Activation('softmax', name="softmax")(x42)

	if layer==-1:
		model = Model(inputs=inp, outputs=[x3, x6, x11, x14, x19, x22, x27, x30, x36, x40, x43])
	elif layer==0:
		model = Model(inputs=inp, outputs=[x3, x43])
	elif layer==1:
		model = Model(inputs=inp, outputs=[x6, x43])
	elif layer==2:
		model = Model(inputs=inp, outputs=[x11, x43])
	elif layer==3:
		model = Model(inputs=inp, outputs=[x14, x43])
	elif layer==4:
		model = Model(inputs=inp, outputs=[x19, x43])
	elif layer==5:
		model = Model(inputs=inp, outputs=[x22, x43])
	elif layer==6:
		model = Model(inputs=inp, outputs=[x27, x43])

	elif layer==7:
		model = Model(inputs=inp, outputs=[x30, x43])

	elif layer==8:
		model = Model(inputs=inp, outputs=[x36, x43])

	elif layer==9:
		model = Model(inputs=inp, outputs=[x40, x43])
	else:
		model = Model(inputs=inp, outputs=[x43])

	opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# model.summary()
	model.load_weights(filepath)

	return model


names = ['block1_adv_act1', 'block1_adv_act2', 'block2_act1', 'block2_act2', 'block3_act1', 'block3_act2', 'block4_act1', 
		  'block4_act2','fc1_act',  'fc2_act', 'softmax']

model_path = "/media/SATA_1/thilini_open_extra/final_codes/OpenMax/AWF/closed/AWF_closed_keras.h5py"
datapath = "/media/SATA_1/thilini_open_extra/final_datasets/AWF/"

# # ---------------------------------------get_layer_out X
# # datapath = "/home/sec-user/thilini/open-set_LCN/Tor/dataset/"


# X, y, _, _, _, _ = LoadDataNoDefCW()
# X, y = get_small_set(X, y, 150)

# X = X[:, 0:1500, np.newaxis]

# # model_f = load_model(model_path)
# # pred = model_f.predict(X)
# # np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/pred_train', pred)

# for i in range(0, len(names)-1):

# 	m = build(n_closed, model_path, i)
# 	out, _ = m.predict(X)
# 	print(i, out.shape)	
# 	np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/'+names[i], out)

# m = build(n_closed, model_path, len(names)-1)
# out = m.predict(X)
# print(len(names)-1, out.shape)	
# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/'+names[len(names)-1], out)

# # -----------------------------------------------------------------------------get_layer_out open
# X = np.load(datapath+'X_open.npy')
# # y = np.load(datapath+'y_open.npy')

# X = X[0:40000, 0:1500, np.newaxis]
# print(X.shape)

# for i in range(0, len(names)-1):

# 	m = build(n_closed, model_path, i)
# 	out, _ = m.predict(X)
# 	print(out.shape)
# 	np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/open_'+names[i], out)

# m = build(n_closed, model_path, len(names)-1)
# out = m.predict(X)
# print(out.shape)
# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/open_'+names[len(names)-1], out)

# # -----------------------------------------------------------------------------get_layer_out valid
# # datapath =  "/home/sec_user/thilini/Anchor_loss/Tor/dataset/DF_dataset/dataset/"

# _, _, X, y, _, _ = LoadDataNoDefCW()
# # X, y = get_small_set(X, y, 100)

# X = X[:, 0:1500, np.newaxis]
# print(X.shape)

# for i in range(0, len(names)-1):

# 	m = build(n_closed, model_path, i)
# 	out, _ = m.predict(X)
# 	print(out.shape)
# 	np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/valid_'+names[i], out)

# m = build(n_closed, model_path, len(names)-1)
# out = m.predict(X)
# print(out.shape)
# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/valid_'+names[len(names)-1], out)

# # -----------------------------------------------------------------------------get_layer_out test
# # datapath =  "/home/sec_user/thilini/Anchor_loss/Tor/dataset/DF_dataset/dataset/"

# _, _, _, _, X, y = LoadDataNoDefCW()
# X, y = get_small_set(X, y, 150)

# X = X[:, 0:1500, np.newaxis]
# print(X.shape)

# for i in range(0, len(names)-1):

# 	m = build(n_closed, model_path, i)
# 	out, _ = m.predict(X)
# 	print(out.shape)
# 	np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/test_'+names[i], out)

# m = build(n_closed, model_path,  len(names)-1)
# out = m.predict(X)
# print(out.shape)
# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/test_'+names[ len(names)-1], out)


# # # ------------------------------------------get MAVs
# print('Getting MAVs')
# X, y, _, _, _, _ = LoadDataNoDefCW()
# X, y = get_small_set(X, y, 150)

# # outs= []
# # for i in range(0, len(names)):
# # 	outs.append(np.load('/media/SATA_1/thilini_open_extra/AWF_temp/'+names[i]+'.npy'))
# # 	print(outs[i].shape)

# # mavs = np.zeros((n_closed, n_closed+max_positions.shape[0]))


# for c in range(0, n_closed):
# 	idx_c = np.where(y==c)[0]
# 	# print('mav: ', c, len(idx_c))

# 	m = build(n_closed, model_path, 11)
# 	out = m.predict(X[idx_c, 0:1500, np.newaxis])

# 	logits = np.mean(out, axis=0)	

# 	np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/act_only/MAVS/_MAVs_'+str(n_max)+str(c), logits)




# # --------------------------------------------------------------threshs
# _, _, X, y, _, _ = LoadDataNoDefCW()
# # X, y = get_small_set(X, y, 100)
# model_f = load_model(model_path)
# pred = model_f.predict(X[:, :, np.newaxis])
# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/pred_valid', pred)
# pred = np.argmax(pred, axis=1)

# # outs= []
# # for i in range(0, len(names)):
# # 	outs.append(np.load('/media/SATA_1/thilini_open_extra/AWF_temp/valid_'+names[i]+'.npy'))

# # mavs = np.load('/home/sec-user/thilini/open-set_LCN/Tor/layer_out/temp_prenult/_MAVs_'+str(n_max)+'.npy')

# threshs = np.zeros((n_closed,))
# m_prenult = build(n_closed, model_path, 11)

# for c in range(0, n_closed):
# 	print(c)
# 	idx_c = np.where(y==c)[0]
# 	res = []

# 	for i in idx_c:#idx_c
# 		mav = m_prenult.predict(X[i].reshape([1, X.shape[1], 1]))
		
# 		a=np.array(mav)
# 		# b = mavs[int(y[i])]
# 		b = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/act_only/MAVS/_MAVs_'+str(n_max)+str(c)+'.npy', allow_pickle=True)
# 		diff = a-b
# 		diff = np.linalg.norm(diff)
# 		# diff = np.dot(a, b)/( np.linalg.norm(a)* np.linalg.norm(b))
# 		# print(diff.shape)
# 		res.append(diff)
# 	res = np.array(res)
# 	start = np.max(res)
# 	end = np.min(res)
# 	gap = -0.02 #(end- start)/200000 # precision:200000

# 	accuracy_thresh = 90.0 
# 	accuracy_range = np.arange(start, end, gap)
# 	for i, delta in enumerate(accuracy_range):
# 		Inliers_label = np.where(res<=delta, pred[idx_c], n_closed)
# 		y_ = y[idx_c]
# 		# print(Inliers_label.shape, y_.shape)
# 		a = np.sum(np.where(y_ == Inliers_label, 1, 0))/Inliers_label.shape[0]*100  

# 		if i==0 and a<accuracy_thresh:
# 			print('Closed set accuracy did not reach ', accuracy_thresh, a)
# 			threshs[c] = delta
# 			break

# 		elif a<accuracy_thresh and i>0:
# 			delta = accuracy_range[i-1]
# 			threshs[c] = delta
# 			print('ideal, thresh:{}, prev_acc:{}'.format(delta, a))
# 			break

# 		elif i==len(accuracy_range)-1:
# 			print('Closed set accuracy did not fall below ', accuracy_thresh, a)
# 			threshs[c] = delta
# 			break

# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/act_only/threshs_list_'+str(n_max), threshs)




# -------------------------------------------------------------tets closed
_, _, _, _, X, y, = LoadDataNoDefCW()
X, y = get_small_set(X, y, 150)
# X = np.load(datapath+'X_test.npy')[:, 0:1500]
# y = np.load(datapath+'y_test.npy')

# model_f = load_model(model_path)
# pred = model_f.predict(X[:, :, np.newaxis])
# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/pred_test', pred)
pred = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/pred_test.npy')
pred = np.argmax(pred, axis=1)
res = []

# outs= []
# for i in range(0, len(names)):
# 	# print(i)
# 	a = np.load('/media/SATA_1/thilini_open_extra/AWF_temp/test_'+names[i]+'.npy')
# 	print(a.shape)
# 	outs.append(a)
# 	# outs.append(np.load('/home/sec-user/thilini/open-set_LCN/Tor/layer_out/test_'+names[i]+'.npy'))

# mavs = np.load('/home/sec-user/thilini/open-set_LCN/Tor/layer_out/temp_prenult/_MAVs_'+str(n_max)+'.npy')
threshs = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/act_only/threshs_list_'+str(n_max)+'.npy')
m_prenult = build(n_closed, model_path, 11)
probs = []

for c in range(0, 1):

	for i in range(0, X.shape[0]):
		mav = m_prenult.predict(X[i].reshape([1, X.shape[1], 1]))
		a = mav
		# b = mavs[int(y[i])]
		b = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/act_only/MAVS/_MAVs_'+str(n_max)+str(int(pred[i]))+'.npy')
		diff = a - b
		diff = np.linalg.norm(diff)
		# diff = np.dot(a, b)/( np.linalg.norm(a)* np.linalg.norm(b))
		if diff>threshs[int(pred[i])]:
			pred[i] = n_closed

		probs.append(diff)

res = np.where(pred==y, 1, 0)
print('closed {}'.format(np.sum(res)/res.shape[0]*100))
prob_c = np.array(probs)
pred_c = pred
del res, X

# # -------------------------------------------------------------tets open
l = 30000

# # outs= []
# # for i in range(0, len(names)):
# # 	# print(i)
# # 	a = np.load('/media/SATA_1/thilini_open_extra/AWF_temp/open_'+names[i]+'.npy')[0:l]
# # 	print(a.shape)
# # 	outs.append(a)
# # print('predicted')

X = np.load(datapath+'X_open.npy')[0:l:, 0:1500]

model_f = load_model(model_path)
# pred = model_f.predict(X[:, :, np.newaxis])
# np.save('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/pred_open', pred)
pred = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/pred_open.npy')
pred = np.argmax(pred, axis=1)
probs = []


# mavs = np.load('/home/sec-user/thilini/open-set_LCN/Tor/layer_out/temp_prenult/_MAVs_'+str(n_max)+'.npy')
threshs = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/act_only/threshs_list_'+str(n_max)+'.npy')
m_prenult = build(n_closed, model_path, 11)
print('-------------------          predicted')

for i in range(0, X.shape[0]):	
	# print(i)
	mav = m_prenult.predict(X[i].reshape([1, X.shape[1], 1]))
	a = mav
	# b = mavs[int(y[i])]
	b = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/AWF/temp/act_only/MAVS/_MAVs_'+str(n_max)+str(int(pred[i]))+'.npy')
	diff = a - b
	diff = np.linalg.norm(diff)
	# diff = np.dot(a, b)/( np.linalg.norm(a)* np.linalg.norm(b))
	if diff>threshs[int(pred[i])]:
		pred[i] = n_closed
	# if i%100==0:
	# 	print(i)
	probs.append(diff)

pred = pred[0:l]
y_o = np.ones((pred.shape[0],))*n_closed

res = np.where(pred==y_o, 1, 0)
print('open {}'.format(np.sum(res)/res.shape[0]*100))
del X, res
prob_o = np.array(probs)
pred_o = pred

auroc_= auroc(prob_c, prob_o, title='test', trial_num=1)
f1 = get_f1(y, y_o, pred_c, pred_o, n_closed)

print('AUROC: {}'.format(auroc_*100))
print('F1:{}'.format(f1))