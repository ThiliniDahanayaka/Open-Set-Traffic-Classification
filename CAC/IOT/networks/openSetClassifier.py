"""
	Network definition for our proposed CAC open set classifier. 

	Dimity Miller, 2020
"""


import torch
import torchvision
import torch.nn as nn

class GlobalAVGPooling1D(nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalAVGPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.mean(input, axis=self.step_axis).values

class openSetClassifier(nn.Module):
	def __init__(self, num_classes = 30, init_weights = False, **kwargs):
		super(openSetClassifier, self).__init__()

		self.num_classes = num_classes
		self.encoder = BaseEncoder(init_weights)
		
		# if im_size == 32:
		# 	self.classify = nn.Linear(128*4*4, num_classes)
		# elif im_size == 64:
		# 	self.classify = nn.Linear(128*8*8, num_classes)
		# else:
		# 	print('That image size has not been implemented, sorry.')
		# 	exit()
		self.classify = nn.Linear(150, num_classes)

		self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad = False)

		# if init_weights:
		# 	self._initialize_weights()
		
		self.cuda()


	def forward(self, x, skip_distance = False):
		batch_size = len(x)

		x = self.encoder(x)
		x = x.view(batch_size, -1)

		outLinear = self.classify(x)

		if skip_distance:
			return outLinear, None

		outDistance = self.distance_classifier(outLinear)

		return outLinear, outDistance

	# def _initialize_weights(self):
	# 	for m in self.modules():
	# 		if isinstance(m, nn.Conv2d):
	# 			nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
	# 			if m.bias is not None:
	# 				nn.init.constant_(m.bias, 0)
	# 		elif isinstance(m, nn.BatchNorm2d):
	# 			nn.init.constant_(m.weight, 1)
	# 			nn.init.constant_(m.bias, 0)
	# 		elif isinstance(m, nn.Linear):
	# 			nn.init.normal_(m.weight, 0, 0.01)
	# 			nn.init.constant_(m.bias, 0)

	def set_anchors(self, means):
		self.anchors = nn.Parameter(means.double(), requires_grad = False)
		self.cuda()

	def distance_classifier(self, x):
		''' Calculates euclidean distance from x to each class anchor
			Returns n x m array of distance from input of batch_size n to anchors of size m
		'''

		n = x.size(0)
		m = self.num_classes
		d = self.num_classes

		x = x.unsqueeze(1).expand(n, m, d).double()
		anchors = self.anchors.unsqueeze(0).expand(n, m, d)
		dists = torch.norm(x-anchors, 2, 2)

		return dists

class BaseEncoder(nn.Module):
    def __init__(self, init_weights, **kwargs): 
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, stride=1, padding='valid', bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=19, stride=1, padding='valid')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=13, stride=1, padding='valid')
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=23, stride=1, padding='valid')
        

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(180)
        self.bn6 = nn.BatchNorm1d(150)

        self.maxpool1 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.maxpool4 = nn.MaxPool1d(kernel_size=1, stride=1)

        self.avgpool = GlobalAVGPooling1D()
        self.flat = nn.Flatten()

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.1)

        self.elu = nn.ELU(alpha=1.0)
        self.tanh = nn.Tanh()
        self.selu = nn.SELU()

        self.dense1 = nn.Linear(106752, 180)
        self.dense2 = nn.Linear(180, 150)


        self.encoder1 = nn.Sequential(
                            self.conv1,
                            self.tanh,
                            self.bn1,
                            self.maxpool1,
                            self.dropout1,
                            self.conv2,
                            self.elu,
                            self.bn2,
                            self.maxpool2,
                            self.dropout2,
                            self.conv3,
                            self.elu,
                            self.bn3,
                            self.maxpool3,
                            self.dropout3,
                        )

        self.encoder2 = nn.Sequential(
                            self.conv4,
                            self.selu,
                            self.bn4,
                            self.maxpool4,
                            self.flat,
                            )

        self.encoder3 = nn.Sequential(
                                self.dense1,
                                self.selu,
                                self.bn5,
                                self.dense2,
                                self.selu,
                                self.bn6,
                            )

        

        # if init_weights:
        #     self._initialize_weights()
    
        self.cuda()


    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        return x3







