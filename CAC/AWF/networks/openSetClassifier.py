"""
	Network definition for our proposed CAC open set classifier. 

	Dimity Miller, 2020
"""


import torch
import torchvision
import torch.nn as nn

class openSetClassifier(nn.Module):
	def __init__(self, num_classes = 20, init_weights = False, **kwargs):
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
		self.classify = nn.Linear(512, num_classes)

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

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding='same')
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='same')
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding='same')
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding='same')
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1, padding='same')
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8, stride=1, padding='same')

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(256)

        self.elu = nn.ELU(alpha=1.0)

        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)

        self.dropout = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()

        self.flat = nn.Flatten()

        self.dense1 = nn.Linear(1280, 512)
        self.dense2 = nn.Linear(512, 512)

        self.bn9 = nn.BatchNorm1d(512)
        self.bn10 = nn.BatchNorm1d(512)

        self.drop1 = nn.Dropout(0.7)
        self.drop2 = nn.Dropout(0.5)


        self.encoder1 = nn.Sequential(
                            self.conv1,
                            self.bn1,
                            self.elu,
                            self.conv2,
                            self.bn2,
                            self.elu,
                            self.maxpool,
                            self.dropout,
                        )

        self.encoder2 = nn.Sequential(
                                self.conv3,
                                self.bn3,
                                self.relu,
                                self.conv4,
                                self.bn4,
                                self.relu,
                                self.maxpool,
                                self.dropout,
                            )

        self.encoder3 = nn.Sequential(
                                self.conv5,
                                self.bn5,
                                self.relu,
                                self.conv6,
                                self.bn6,
                                self.relu,
                                self.maxpool,
                                self.dropout,
                            )

        self.encoder4 = nn.Sequential(
                                self.conv7,
                                self.bn7,
                                self.relu,
                                self.conv8,
                                self.bn8,
                                self.relu,
                                self.maxpool,
                                self.dropout,
                            )

        self.encoder5 = nn.Sequential(
                                self.flat, 
                                self.dense1, 
                                self.bn9,
                                self.relu,
                                self.drop1,
                                self.dense2, 
                                self.bn10,
                                self.relu,
                                self.drop2,
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
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        return x5







