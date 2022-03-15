
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from itertools import product



# Draw confusion matrix
def plot_confusion_matrice(x, labels, name, n_classes):
	sample_weight=None
	normalize=None
	display_labels=None
	include_values=True
	xticks_rotation='vertical'
	values_format=None
	cmap='Reds'#'viridis'
	ax=None


	if ax is None:
		fig, ax = plt.subplots(figsize=(50,50))
	else:
		fig = ax.figure
	# plt.rcParams['axes.labelsize'] = 30
	cm = x

	im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

	if include_values:
		text_ = np.empty_like(cm, dtype=object)
	if values_format is None:
		values_format = '.1f'

    # print text with appropriate color depending on background
	thresh = (cm.max() + cm.min()) / 2.0
	for i, j in product(range(n_classes), range(n_classes)):
		color = cmap_max if cm[i, j] < thresh else cmap_min
		text_[i, j] = ax.text(j, i, format(cm[i, j], values_format), ha="center", va="center", color=color, fontsize=40)  
	cbar = fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.04)
	# cbar =fig.colorbar.ax.tick_params(labelsize=20)
	cbar.ax.tick_params(labelsize=30)

	ax.set(xticks=np.arange(0, n_classes),
        yticks=np.arange(0, n_classes),
        xticklabels=labels, 
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label")
 
	ax.set_ylim((n_classes - 0.5, -0.5))
	plt.setp(ax.get_xticklabels(), rotation=xticks_rotation, fontsize=30)
	plt.setp(ax.get_yticklabels(), fontsize=30)
	ax.xaxis.label.set_size(30)
	ax.yaxis.label.set_size(30)


	plt.savefig(name, bbox_inches = 'tight')
	plt.close()