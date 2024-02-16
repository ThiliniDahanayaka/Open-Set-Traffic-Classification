import numpy as np
import pandas
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.spatial
import libmr


class Openmax():
    def __init__(self, decision_dist_fn='euclidean', tailsize=20, alpharank=1):
        """
        Args:
            z: 3D numpy array---(n,,),the last second layer, activate vector
            softmax_layer: 3D numpy array---(n,,n_classes),the last second layer
            labels: 2D or 1D numpy array --(n,)
            n_classes: `int`, number of classes - 10 or 100

        """

        self.decision_dist_fn = decision_dist_fn
        self.tailsize = tailsize
        self.alpharank = alpharank

    def dist_from_mav(self, Z, c_mu):
        if self.decision_dist_fn == 'euclidean':
            return scipy.spatial.distance.cdist(Z, c_mu, metric=self.decision_dist_fn) / 200
        elif self.decision_dist_fn == 'eucos':
            return (scipy.spatial.distance.cdist(Z, c_mu, metric='euclidean') / 200) + \
                   scipy.spatial.distance.cdist(Z, c_mu, metric='cosine')
        elif self.decision_dist_fn == 'cos':
            return scipy.spatial.distance.cdist(Z, c_mu, metric='cosine')

    def update_class_stats(self, z, pred_y, y):
        self.mr_model = {}
        self.c_means = np.zeros((y.shape[1], z.shape[1]))

        correct = (np.argmax(pred_y, axis=1) == np.argmax(y, axis=1))
        z = z[correct]
        pred_y = pred_y[correct]

        # fit weibull model for each class
        
        for c in range(y.shape[1]):
            # Calculate Class Mean
            # print(c)
            z_c = z[np.argmax(pred_y, axis=1) == c]
            if z_c.shape[0]>1:
                mu_c = z_c.mean(axis=0)

                # Fit Weibull
                mr = libmr.MR()
                # tailtofit = sorted(self.dist_from_mav(z_c, mu_c[None, :]).ravel())[int(-self.tailsizea[c]):]
                tailtofit = sorted(self.dist_from_mav(z_c, mu_c[None, :]).ravel())[-self.tailsize:]
                mr.fit_high(tailtofit, len(tailtofit))
                # assert len(str(mr)) > 0
                self.mr_model[c] = mr
                self.c_means[c, :] = mu_c
            else:
                self.mr_model[c] = -1
                self.c_means[c, :] = -1

    def normalize_f(self, x):
        m_ax = np.max(x, axis=1)
        m_ax = np.reshape(m_ax, [m_ax.shape[0], 1])
        m_in = np.min(x, axis=1)
        m_in = np.reshape(m_in, [m_in.shape[0], 1])

        s = (x - m_in)/(m_ax - m_in)
        return s

    def predict_prob_open(self, logits):
        alpha_weights = [((self.alpharank + 1) - i) / float(self.alpharank) for i in range(1, self.alpharank + 1)]
        descending_argsort = np.fliplr(np.argsort(logits, axis=1))
        logits_normalized = np.zeros((logits.shape[0], logits.shape[1] + 1))

        fixed_logits = self.normalize_f(logits)

        all_dist = self.dist_from_mav(logits, self.c_means)
        for i in range(logits.shape[0]):  # for each data point
            for alpha in range(self.alpharank):
                c = descending_argsort[i, alpha]
                # works only when alpha=1
                # logit_val = abs(logits[i, c])
                logit_val = fixed_logits[i, c]
                if (self.mr_model[c]==-1):
                    logits_normalized[i, c] = 0
                    logits_normalized[i, -1] = 1
                else:
                    ws_c = 1 - self.mr_model[c].w_score(all_dist[i, c]) * alpha_weights[alpha]
                    logits_normalized[i, c] = logit_val * ws_c
                    logits_normalized[i, -1] += logit_val * (1 - ws_c)
        open_prob = np.exp(logits_normalized)
        if np.any(open_prob.sum(axis=1)[:, None] == np.inf):
            print('Error: Inf value has been returned from w_score function. Consider training with larger '
                  'tailsize value.')
        open_prob = open_prob / open_prob.sum(axis=1)[:, None]
        return open_prob















# import numpy as np
# import pandas
# import os
# import sys
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# import scipy.spatial
# import libmr


# class Openmax():
#     def __init__(self, decision_dist_fn='euclidean', tailsize=20, alpharank=1):
#         """
#         Args:
#             z: 3D numpy array---(n,,),the last second layer, activate vector
#             softmax_layer: 3D numpy array---(n,,n_classes),the last second layer
#             labels: 2D or 1D numpy array --(n,)
#             n_classes: `int`, number of classes - 10 or 100

#         """
#         highest = np.array([64, 57, 35, 40, 73, 0, 60, 36, 43, 50])
#         next_highest = np.array([8, 2 ,3, 79, 63, 69, 61, 65])
#         a = np.ones((86,))*25
#         a[highest] = a[highest]*4
#         a[next_highest] = a[next_highest]*2
#         self.tailsizea = a
#         self.decision_dist_fn = decision_dist_fn
#         self.tailsize = tailsize
#         self.alpharank = alpharank

#     def dist_from_mav(self, Z, c_mu):
#         if self.decision_dist_fn == 'euclidean':
#             return scipy.spatial.distance.cdist(Z, c_mu, metric=self.decision_dist_fn) / 200
#         elif self.decision_dist_fn == 'eucos':
#             return (scipy.spatial.distance.cdist(Z, c_mu, metric='euclidean') / 200) + \
#                    scipy.spatial.distance.cdist(Z, c_mu, metric='cosine')
#         elif self.decision_dist_fn == 'cos':
#         	return scipy.spatial.distance.cdist(Z, c_mu, metric='cosine')

#     def update_class_stats(self, z, pred_y, y):
#         correct = (np.argmax(pred_y, axis=1) == np.argmax(y, axis=1))
#         z = z[correct]
#         pred_y = pred_y[correct]

#         # fit weibull model for each class
#         self.mr_model = {}
#         self.c_means = np.zeros((y.shape[1], z.shape[1]))
#         for c in range(y.shape[1]):
#             # Calculate Class Mean
#             # print(c)
#             z_c = z[np.argmax(pred_y, axis=1) == c]
#             # print(c, z_c.shape)
#             mu_c = z_c.mean(axis=0)

#             # Fit Weibull
#             mr = libmr.MR()
#             # tailtofit = sorted(self.dist_from_mav(z_c, mu_c[None, :]).ravel())[int(-self.tailsizea[c]):]
#             tailtofit = sorted(self.dist_from_mav(z_c, mu_c[None, :]).ravel())[-self.tailsize:]
#             mr.fit_high(tailtofit, len(tailtofit))
#             # assert len(str(mr)) > 0
#             self.mr_model[c] = mr
#             self.c_means[c, :] = mu_c

#     def normalize_f(self, x):
#         m_ax = np.max(x, axis=1)
#         m_ax = np.reshape(m_ax, [m_ax.shape[0], 1])
#         m_in = np.min(x, axis=1)
#         m_in = np.reshape(m_in, [m_in.shape[0], 1])

#         s = (x - m_in)/(m_ax - m_in)
#         return s

#     def predict_prob_open(self, logits):
#         alpha_weights = [((self.alpharank + 1) - i) / float(self.alpharank) for i in range(1, self.alpharank + 1)]
#         descending_argsort = np.fliplr(np.argsort(logits, axis=1))
#         logits_normalized = np.zeros((logits.shape[0], logits.shape[1] + 1))

#         fixed_logits = self.normalize_f(logits)

#         all_dist = self.dist_from_mav(logits, self.c_means)
#         for i in range(logits.shape[0]):  # for each data point
#             for alpha in range(self.alpharank):
#                 c = descending_argsort[i, alpha]
#                 # works only when alpha=1
#                 # logit_val = abs(logits[i, c])
#                 logit_val = fixed_logits[i, c]
#                 ws_c = 1 - self.mr_model[c].w_score(all_dist[i, c]) * alpha_weights[alpha]
#                 logits_normalized[i, c] = logit_val * ws_c
#                 logits_normalized[i, -1] += logit_val * (1 - ws_c)
#         open_prob = np.exp(logits_normalized)
#         if np.any(open_prob.sum(axis=1)[:, None] == np.inf):
#             print('Error: Inf value has been returned from w_score function. Consider training with larger '
#                   'tailsize value.')
#         open_prob = open_prob / open_prob.sum(axis=1)[:, None]
#         return open_prob



# # class Openmax():
# #     def __init__(self, decision_dist_fn='euclidean', tailsize=20, alpharank=1):
# #         """
# #         Args:
# #             z: 3D numpy array---(n,,),the last second layer, activate vector
# #             softmax_layer: 3D numpy array---(n,,n_classes),the last second layer
# #             labels: 2D or 1D numpy array --(n,)
# #             n_classes: `int`, number of classes - 10 or 100

# #         """
# #         highest = np.array([64, 57, 35, 40, 73, 0, 60, 36, 43, 50])
# #         next_highest = np.array([8, 2 ,3, 79, 63, 69, 61, 65])
# #         a = np.ones((86,))*25
# #         a[highest] = a[highest]*4
# #         a[next_highest] = a[next_highest]*2
# #         self.tailsizea = a
# #         self.decision_dist_fn = decision_dist_fn
# #         self.tailsize = tailsize
# #         self.alpharank = alpharank

# #     def dist_from_mav(self, Z, c_mu):
# #         if self.decision_dist_fn == 'euclidean':
# #             return scipy.spatial.distance.cdist(Z, c_mu, metric=self.decision_dist_fn) / 200
# #         elif self.decision_dist_fn == 'eucos':
# #             return (scipy.spatial.distance.cdist(Z, c_mu, metric='euclidean') / 200) + \
# #                    scipy.spatial.distance.cdist(Z, c_mu, metric='cosine')
# #         elif self.decision_dist_fn == 'cos':
# #             return scipy.spatial.distance.cdist(Z, c_mu, metric='cosine')

# #     def update_class_stats(self, z, pred_y, y):
# #         self.mr_model = {}
# #         self.c_means = np.zeros((y.shape[1], z.shape[1]))

# #         correct = (np.argmax(pred_y, axis=1) == np.argmax(y, axis=1))
# #         z = z[correct]
# #         pred_y = pred_y[correct]

# #         # fit weibull model for each class
        
# #         for c in range(y.shape[1]):
# #             # Calculate Class Mean
# #             # print(c)
# #             z_c = z[np.argmax(pred_y, axis=1) == c]
# #             if z_c.shape[0]>1:
# #                 mu_c = z_c.mean(axis=0)

# #                 # Fit Weibull
# #                 mr = libmr.MR()
# #                 # tailtofit = sorted(self.dist_from_mav(z_c, mu_c[None, :]).ravel())[int(-self.tailsizea[c]):]
# #                 tailtofit = sorted(self.dist_from_mav(z_c, mu_c[None, :]).ravel())[-self.tailsize:]
# #                 mr.fit_high(tailtofit, len(tailtofit))
# #                 # assert len(str(mr)) > 0
# #                 self.mr_model[c] = mr
# #                 self.c_means[c, :] = mu_c
# #             else:
# #                 self.mr_model[c] = -1
# #                 self.c_means[c, :] = -1

# #     def normalize_f(self, x):
# #         m_ax = np.max(x, axis=1)
# #         m_ax = np.reshape(m_ax, [m_ax.shape[0], 1])
# #         m_in = np.min(x, axis=1)
# #         m_in = np.reshape(m_in, [m_in.shape[0], 1])

# #         s = (x - m_in)/(m_ax - m_in)
# #         return s

# #     def predict_prob_open(self, logits):
# #         alpha_weights = [((self.alpharank + 1) - i) / float(self.alpharank) for i in range(1, self.alpharank + 1)]
# #         descending_argsort = np.fliplr(np.argsort(logits, axis=1))
# #         logits_normalized = np.zeros((logits.shape[0], logits.shape[1] + 1))

# #         fixed_logits = self.normalize_f(logits)

# #         all_dist = self.dist_from_mav(logits, self.c_means)
# #         for i in range(logits.shape[0]):  # for each data point
# #             for alpha in range(self.alpharank):
# #                 c = descending_argsort[i, alpha]
# #                 # works only when alpha=1
# #                 # logit_val = abs(logits[i, c])
# #                 logit_val = fixed_logits[i, c]
# #                 if (self.mr_model[c]==-1):
# #                     logits_normalized[i, c] = 0
# #                     logits_normalized[i, -1] = 1
# #                 else:
# #                     ws_c = 1 - self.mr_model[c].w_score(all_dist[i, c]) * alpha_weights[alpha]
# #                     logits_normalized[i, c] = logit_val * ws_c
# #                     logits_normalized[i, -1] += logit_val * (1 - ws_c)
# #         open_prob = np.exp(logits_normalized)
# #         if np.any(open_prob.sum(axis=1)[:, None] == np.inf):
# #             print('Error: Inf value has been returned from w_score function. Consider training with larger '
# #                   'tailsize value.')
# #         open_prob = open_prob / open_prob.sum(axis=1)[:, None]
# #         return open_prob
