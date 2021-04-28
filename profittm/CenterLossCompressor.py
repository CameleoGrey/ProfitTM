import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.optim as optim

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from profittm.save_load import *

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CenterLossNN(nn.Module):
    def __init__(self, x_shape, nClasses, latentDim):
        super(CenterLossNN, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=2)
        torch.nn.init.kaiming_normal_(self.conv1_1.weight)
        self.bn_1 = nn.BatchNorm2d(32)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=2)
        torch.nn.init.kaiming_normal_(self.conv1_2.weight)
        self.bn_2 = nn.BatchNorm2d(64)
        self.do_1 = nn.Dropout2d(p=0.2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=2)
        torch.nn.init.kaiming_normal_(self.conv2_1.weight)
        self.bn_3 = nn.BatchNorm2d(128)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*(x_shape[2]-3), latentDim)
        self.ip2 = nn.Linear(latentDim, nClasses, bias=True)

    def forward(self, x):
        x = self.prelu1_1(self.bn_1(self.conv1_1(x)))
        x = self.prelu1_2(self.do_1(self.bn_2(self.conv1_2(x))))
        x = self.prelu3_2(self.bn_3(self.conv2_1(x)))
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2, dim=1)

class CenterLossCompressor():
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda")


        self.latent_dim = None
        self.nClasses = None

        pass

    def fit(self, x, y, validation_part=0.05, batchSize=100, epochs=100, latent_dim=100):

        self.nClasses = len(np.unique(y))
        self.latent_dim = latent_dim

        self.model = CenterLossNN(x_shape=x.shape, nClasses=self.nClasses, latentDim=latent_dim)
        self.model.to(self.device)

        loss_weight = 1
        nllloss = nn.CrossEntropyLoss().to(self.device)
        centerloss = CenterLoss(self.nClasses, self.latent_dim).to(self.device)
        optimizer4nn = optim.Adam(self.model.parameters(), lr=0.001)
        optimzer4center = optim.Adam(centerloss.parameters(), lr=0.5)

        for epoch in range(epochs):
            train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = validation_part)

            nBatches = len(train_y) // batchSize
            if nBatches == 0:
                nBatches = 1
            batchesX = np.array_split(train_x, nBatches)
            batchesY = np.array_split(train_y, nBatches)

            for i in tqdm(range(len(batchesX)), desc="Epoch {} of {} | Batches processed".format(epoch, epochs)):
                data = batchesX[i]
                data = torch.Tensor(data)
                data = torch.unsqueeze(data, dim=1)
                target = batchesY[i]
                target = torch.tensor(target, dtype=torch.int64)
                data = data.to(self.device)
                target = target.to(self.device)

                ip1, pred = self.model(data)
                loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1)
                if i % (1 + nBatches // 10) == 0:
                    print(" Train loss: {}".format(loss.data.cpu().numpy()[0]))

                optimizer4nn.zero_grad()
                optimzer4center.zero_grad()
                loss.backward()
                optimizer4nn.step()
                optimzer4center.step()
        torch.cuda.empty_cache()
        pass

    def predict(self, x, batchSize=100):

        nBatches = len(x) // batchSize
        if nBatches == 0:
            nBatches = 1
        batchesX = np.array_split(x, nBatches)

        features = []
        for batch in batchesX:
            with torch.no_grad():
                batch = torch.Tensor(batch)
                batch = torch.unsqueeze(batch, dim=1)
                batch = batch.to(self.device)
                feats, labels = self.model(batch)
                feats = feats.data.cpu().numpy()
                features.append(feats)
                batch = batch.to("cpu")
                del batch
        features = np.vstack(features)
        torch.cuda.empty_cache()

        return features

    def saveFeatureExtractor(self, name="bung", path="./"):
        save(path + name + ".pkl", self)
        return self

    def loadFeatureExtractor(self, name="bung", path="./"):
        loadedFeatExtractor = load(path + name + ".pkl")
        return loadedFeatExtractor