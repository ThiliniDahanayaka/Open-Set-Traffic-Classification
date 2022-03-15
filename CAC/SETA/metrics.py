"""
	Metrics used to evaluate performance.

	Dimity Miller, 2020
"""
import numpy as np
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
#from conf_mat_plot import plot_confusion_matrice


# def confusion_matrix(pred, label, n_closed):
#     mat = np.zeros((n_closed+1, n_closed+1))

#     # y-axis: label, x-axis:prediction
#     for i in range(0, pred.shape[0]):
#         # print(int(label[i]), int(pred[i]))
#         mat[int(label[i]), int(pred[i])] = mat[int(label[i]), int(pred[i])] + 1

#     # Uncomment if plotting the confusion matrix
#     count = np.sum(mat, axis=1)
#     count = np.where(count>np.zeros(count.shape), count, 1)
#     for i in range(0, n_closed+1):
#         mat[i, :] = mat[i, :]/count[i]


#     return mat
# def accuracy(x, gt):
# 	predicted = np.argmin(x, axis = 1)
# 	total = len(gt)
# 	acc = np.sum(predicted == gt)/total
# 	return acc


# def auroc(inData, outData, title='test', in_low = True, trial_num=1):
# 	# print(inData.shape, outData.shape)
# 	inDataMin = np.min(inData, 1)
# 	outDataMin = np.min(outData, 1)
# 	# print(inData.max(), outData.max())
	


# 	allData = np.concatenate((inDataMin, outDataMin))
# 	labels = np.concatenate((np.zeros(len(inDataMin)), np.ones(len(outDataMin))))
# 	fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = in_low)

# 	# for k in range(0, len(thresholds)):
# 	# 	print('At thresh {}: TPR={} and FPR={}'.format(thresholds[k], tpr[k], fpr[k]))

# 	plt.plot(fpr, tpr)
# 	plt.xlabel('FPR')
# 	plt.ylabel('TPR')

# 	auc = sklearn.metrics.auc(fpr, tpr)

# 	plt.title(title+' with AUC:{}'.format(auc))
# 	plt.savefig(title+'_{}.png'.format(trial_num))
# 	plt.close()



# 	return sklearn.metrics.auc(fpr, tpr)

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
	

def auroc(inData, outData, title='test', trial_num=1):
	# print(inData.shape, outData.shape)
	inDataMin = np.min(inData, 1)
	outDataMin = np.min(outData, 1)

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

def get_f1(Inliers_true, other_true, Inliers_score_raw,  other_score_raw, delta, num_classes):
	'''
	delta: threshold value to reject open samples calculated from method 'accuracy'
	'''
	Inliers_pred = np.argmin(Inliers_score_raw, axis=1)
	other_pred = np.argmin(other_score_raw, axis=1)

	Inliers_score = np.min(Inliers_score_raw, axis=1)
	other_score = np.min(other_score_raw, axis=1)

	Inliers_label = np.where(Inliers_score<=delta, Inliers_pred, num_classes)
	Outliers_label = np.where(other_score<=delta, other_pred, num_classes)

	prediction = np.append(Inliers_label, Outliers_label, axis=0)
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

def classwiseAccuracy(dist_known, dist_unknown, label_known, label_unknown, title, thresh, trial_num=1):
	pred_knownMin = np.argmin(dist_known, 1)
	pred_unknownMin = np.argmin(dist_unknown, 1)

	dist_knownMin = np.min(dist_known, 1)
	dist_unknownMin = np.min(dist_unknown, 1)

	# t = label_known[np.where(np.isnan(dist_known))]
	# t = np.where(np.isnan(dist_known))
	# t = label_known[0:272]
	# print(np.unique(t))

	# t = label_unknown[np.where(np.isnan(dist_unknown))]
	# print(np.unique(t))

	# print(dist_knownMin.max(), dist_unknownMin.max())
	
	allData = np.concatenate((pred_knownMin, pred_unknownMin))
	all_data2 = np.concatenate((dist_knownMin, dist_unknownMin))
	labels = np.concatenate(( label_known, label_unknown)).astype('int')

	temp = np.where(dist_knownMin<thresh, pred_knownMin,  np.max(labels))
	temp = np.where(temp==label_known, 1, 0)
	print('Closed-set acc:', np.sum(temp)/temp.shape[0]*100, '%')

	temp = np.where(dist_unknownMin<thresh, pred_unknownMin,  np.max(labels))
	temp = np.where(temp==label_unknown, 1, 0)
	print('Open-set acc:', np.sum(temp)/temp.shape[0]*100, '%')

	all_data2 = np.where(all_data2<thresh, allData, np.max(labels))

	conf_mat = confusion_matrix(all_data2, labels, int(labels.max()))

	plot_confusion_matrice(conf_mat, np.arange(0, 5), title+'Conf_mat'+str(trial_num), 5)

	res = np.where(all_data2==labels, 1, 0)
	

	acc = []

	for c in range(0, int(labels.max())+1):

		ind = np.where(labels==c)[0]
		# print(c, len(ind), res.shape[0])
		acc.append(np.sum(res[ind])/len(ind)*100)

	acc = np.array(acc)

	plt.plot(np.arange(0,acc.shape[0] ), acc)
	plt.xlabel('Class')
	plt.ylabel('Accuracy')
	plt.title(title+' classwise accuracy')
	plt.savefig(title+'_classwise accuracy_trail{}.png'.format(trial_num))
	plt.close()




def distribution(dist_known, dist_unknown, label_known, label_unknown, title, trial_num=1):
	# pred_knownMin = np.argmin(dist_known, 1)
	# pred_unknownMin = np.argmin(dist_unknown, 1)

	dist_knownMin = np.min(dist_known, 1)
	dist_unknownMin = np.min(dist_unknown, 1)

	up = max(np.max(dist_knownMin), np.max(dist_unknownMin))
	down = min(np.min(dist_knownMin), np.min(dist_unknownMin))

	_, temp, _= plt.hist(dist_knownMin, label='Closed-set', bins=100, range=(down, up))
	plt.hist(dist_unknownMin, label='Open-set', bins=100, range=(down, up), alpha=0.3)
	plt.legend()
	plt.title(title+' distribution of distance to anchor')
	plt.savefig(title+'_dist_trial_{}.png'.format(trial_num))
	plt.close()
	# print(temp)
	# print(len(temp))



