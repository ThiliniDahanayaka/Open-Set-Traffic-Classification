import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from utility import LoadDataNoDefCW
from keras.utils import np_utils
import sklearn.metrics	

n_closed = 4
n_max= 3

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

		if (tp+fp==0):
			P = P
		else:
			P = P+(tp/(tp+fp))

		if (tp+fn==0):
			R = R
		else:
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
    from keras.layers import Dense, Layer, Input
    from keras.layers import Conv1D, MaxPooling1D
    from keras.layers.core import Activation, Flatten, Dense, Dropout
    from keras.optimizers import Adamax

    filter_num = [None, 32, 32, 32]
    kernel_size = [None, 16, 16, 16]
    conv_stride_size = ['None', 1, 1, 1]
    pool_stride_size = ['None', 6]
    pool_size = ['None', 6]

    length = 500
    input_shape = (length, 1)
    
    inp = Input(shape=(length, 1))
    x1 = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='valid', name='conv1', activation='relu')(inp)
    x2 = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2], strides=conv_stride_size[2], padding='valid',
                         name='conv2', activation='relu')(x1)

    x3 = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3], strides=conv_stride_size[3], padding='valid',
                         name='conv3', activation='relu')(x2)  

    x4 = Dropout(rate=0.5)(x3)
    x5 = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1], padding='valid', name='max_pool')(x4)
    x6 = Activation('relu', name='max_pool_act')(x5)
    x7 = Dropout(rate=0.3)(x6) 

    x8 = Flatten()(x7)

    x9 = Dense(64, activation='relu')(x8)
    x10 = Dropout(rate=0.5)(x9)
    x11 = Dense(n_closed, activation=None)(x10)
    x12 = Activation('softmax')(x11)

    if layer==-1:
        model = Model(inputs=inp, outputs=[x3, x6, x11, x14, x19, x22, x27, x30, x36, x40, x43])
    elif layer==0:
        model = Model(inputs=inp, outputs=[x1, x12])
    elif layer==1:
        model = Model(inputs=inp, outputs=[x2, x12])
    elif layer==2:
        model = Model(inputs=inp, outputs=[x3, x12])
    elif layer==3:
        model = Model(inputs=inp, outputs=[x9, x12])
    # elif layer==11:
    #     model = Model(inputs=inp, outputs=[x11])
    else:
        model = Model(inputs=inp, outputs=[x12])

    opt = 'Adamax'

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# model.summary()
    model.load_weights(filepath)

    return model

# trial = 5
names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
# model_path = "/home/sec_user/thilini/Other_open/openmax/DC/closed/models/dc_closed_trial_"+str(trial)+".hdf5"
temp_fold = "/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/"


# # ------------------------------------------get MAVs
for trial in range(2, 11):
	model_path = "/media/SATA_1/thilini_open_extra/final_codes/OpenMax/DC/closed/models/dc_closed_trial_"+str(trial)+".hdf5"


	max_positions = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/both/max_positions_'+str(n_max)+'_'+str(trial)+'.npy')
	print('Getting MAVs')
	X, y, _, _, _, _, _, _ = LoadDataNoDefCW(str(trial))
	X, y = get_small_set(X, y, 200)

	outs= []
	for i in range(0, len(names)):
		outs.append(np.load(temp_fold+str(trial)+'_'+names[i]+'.npy'))
		print(outs[i].shape)




	for c in range(0, n_closed):
		idx_c = np.where(y==c)[0]

		logits = np.zeros((1,))	
		

		for i in range(0, max_positions.shape[0]):
			idx = max_positions[i]
			layer = int(idx[0])
			row=int(idx[1:5])
			col = int(idx[5:])
			# print('layer:{},  row:{}, col:{}'.format(layer, row, col))

			out = outs[layer][idx_c]

			if out.ndim==3:
				out = out[:, row, col]
			else:
				out = out[:, row]


			out = np.mean(out)
			logits = np.append(logits, out)
		logits = logits[1:]
		np.save('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/neuron_only/MAVS/_MAV'+str(n_max)+'_'+str(trial)+'_'+str(c), logits)



# --------------------------------------------------------------threshs
for trial in range(2, 11):
	model_path = "/media/SATA_1/thilini_open_extra/final_codes/OpenMax/DC/closed/models/dc_closed_trial_"+str(trial)+".hdf5"


	max_positions = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/both/max_positions_'+str(n_max)+'_'+str(trial)+'.npy')

	_, _, X, y, _, _, _, _  = LoadDataNoDefCW(str(trial))
	X, y = get_small_set(X, y, 100)

	model_f = load_model(model_path)
	pred = model_f.predict(X)
	pred = np.argmax(pred, axis=1)

	outs= []
	for i in range(0, len(names)):
		outs.append(np.load(temp_fold+str(trial)+'_valid_'+names[i]+'.npy'))

	threshs = np.zeros((n_closed,))

	for c in range(0, n_closed):
		print(c)
		idx_c = np.where(y==c)[0]
		res = []

		for i in idx_c:#idx_c
			# mav = m_prenult.predict(X[i].reshape([1, X.shape[1], 1]))
			mav = np.zeros((1,))

			for j in range(0,max_positions.shape[0]):
				idx = max_positions[j]
				layer = int(idx[0])
				row=int(idx[1:5])
				col = int(idx[5:])

				out = outs[layer][i]
				# print(c, '**********************',out.shape)

				if out.ndim==2:
					# print(out.shape)
					out = out[row, col]
					# print(out)
				else:
					out = out[row]
				# print(out)
				mav = np.append(mav, out)
			
			a=np.array(mav[1:])
			# b = mavs[int(y[i])]
			b = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/neuron_only/MAVS/_MAV'+str(n_max)+'_'+str(trial)+'_'+str(c)+'.npy', allow_pickle=True)
			diff = a-b
			diff = np.linalg.norm(diff)
			# diff = np.dot(a, b)/( np.linalg.norm(a)* np.linalg.norm(b))
			# print(diff.shape)
			res.append(diff)
		res = np.array(res)
		start = np.max(res)
		end = np.min(res)
		gap = -0.02 #(end- start)/200000 # precision:200000

		accuracy_thresh = 90.0 
		accuracy_range = np.arange(start, end, gap)
		temp_pred = pred[idx_c]
		for i, delta in enumerate(accuracy_range):

			Inliers_label = np.where(res<=delta, temp_pred, n_closed)
			y_ = y[idx_c]
			# print(Inliers_label.shape, y_.shape)
			a = np.sum(np.where(y_ == Inliers_label, 1, 0))/Inliers_label.shape[0]*100  

			if i==0 and a<accuracy_thresh:
				print('Closed set accuracy did not reach ', accuracy_thresh, a)
				threshs[c] = delta
				break

			elif a<accuracy_thresh and i>0:
				delta = accuracy_range[i-1]
				threshs[c] = delta
				print('ideal, thresh:{}, prev_acc:{}'.format(delta, a))
				break

			elif i==len(accuracy_range)-1:
				print('Closed set accuracy did not fall below ', accuracy_thresh, a)
				threshs[c] = delta
				break

	np.save('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/neuron_only/threshs_list_'+str(n_max)+'_'+str(trial), threshs)



# -------------------------------------------------------------tets closed

for trial in range(2, 11):
	model_path = "/media/SATA_1/thilini_open_extra/final_codes/OpenMax/DC/closed/models/dc_closed_trial_"+str(trial)+".hdf5"

	max_positions = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/both/max_positions_'+str(n_max)+'_'+str(trial)+'.npy')
	_, _, _, _, X, y,_, _ = LoadDataNoDefCW(str(trial))
	X, y = get_small_set(X, y, 200)

	model_f = load_model(model_path)
	pred = model_f.predict(X)
	pred = np.argmax(pred, axis=1)
	res = []

	outs= []
	for i in range(0, len(names)):
		# print(i)
		a = np.load(temp_fold+str(trial)+'_test_'+names[i]+'.npy')
		print(a.shape)
		outs.append(a)
		# outs.append(np.load('/home/sec-user/thilini/open-set_LCN/Tor/layer_out/test_'+names[i]+'.npy'))

	threshs = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/neuron_only/threshs_list_'+str(n_max)+'_'+str(trial)+'.npy')
	probs = []

	for c in range(0, 1):

		for i in range(0, X.shape[0]):
			mav = np.zeros((1,))	
			for j in range(0, max_positions.shape[0]):
				idx = max_positions[j]
				layer = int(idx[0])
				row=int(idx[1:5])
				col = int(idx[5:])


				out = outs[layer][i]

				if out.ndim==2:
					out = out[row, col]
				else:
					out = out[row]

				mav = np.append(mav, out)
			a =np.array(mav[1:])
			# b = mavs[int(y[i])]
			b = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/neuron_only/MAVS/_MAV'+str(n_max)+'_'+str(trial)+'_'+str(int(pred[i]))+'.npy')
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
	

	outs= []
	for i in range(0, len(names)):
		# print(i)
		a = np.load(temp_fold+str(trial)+'_open_'+names[i]+'.npy')
		print(a.shape)
		outs.append(a)
		# outs.append(np.load('/home/sec-user/thilini/open-set_LCN/Tor/layer_out/test_'+names[i]+'.npy'))

	_, _, _, _,_, _, X, _ = LoadDataNoDefCW(str(trial))

	model_f = load_model(model_path)
	pred = model_f.predict(X)
	pred = np.argmax(pred, axis=1)
	probs = []


	for i in range(0, X.shape[0]):	
		# print(i)
		mav = np.zeros((1,))	
		for j in range(0, max_positions.shape[0]):
			idx = max_positions[j]
			layer = int(idx[0])
			row=int(idx[1:5])
			col = int(idx[5:])


			out = outs[layer][i]

			if out.ndim==2:
				out = out[row, col]
			else:
				out = out[row]

			mav = np.append(mav, out)

		a =np.array(mav[1:])
		# b = mavs[int(y[i])]
		b = np.load('/media/SATA_1/thilini_open_extra/final_codes/New/DC/temp/neuron_only/MAVS/_MAV'+str(n_max)+'_'+str(trial)+'_'+str(int(pred[i]))+'.npy')
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

