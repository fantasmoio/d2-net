from distutils.command.upload import upload

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------
class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512
        self.use_relu = use_relu
        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output

# ----------------------------------------------------------------------------------------------------------------------
class D2Net(nn.Module):
    def __init__(self, model_file=None, use_relu=True, use_cuda=True):
        super(D2Net, self).__init__()

        # Module 1
        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_relu=use_relu, use_cuda=use_cuda
        )
        #
        # Module 2
        self.detection = HardDetectionModule()
        #
        # Module 3
        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            self.load_state_dict(torch.load(model_file)['model'])

    '''
    def forward(self, batch):
        _, _, h, w = batch.size()
        #
        # Step 1
        dense_features = self.dense_feature_extraction(batch)
        #
        # Step 2
        detections = self.detection(dense_features)
        #
        # Step 3
        #displacements = self.localization(dense_features)
        displacements = RolfsHandcraftedLocalizationModule(dense_features)
        #
        # Result
        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }
'''
# ----------------------------------------------------------------------------------------------------------------------
class HardDetectionAndLocalization:
    def __init__(self):
        pass

    # 2nd and 3rd module combined, avoiding redundant computation
    # This implementation avoids the (sparse) convolution of the
    # original implementation, making it significantly less
    # readable yet faster. If you want to understand what's going
    # on, better look at the classes HardDetectionModule and
    # HandcraftedLocalizationModule
    def detectAndLocalize(self, batch):
        edge_threshold = 5
        threshold = (edge_threshold + 1) ** 2 / edge_threshold

        # INIT - - - - - - - - - - - - - - -
        # dii, di, djj, dj, dij, det
        b, c, h, w = batch.size()
        device = batch.device
        zW = torch.cuda.FloatTensor(b, c, 1, w).fill_(0)
        zH = torch.cuda.FloatTensor(b, c, h, 1).fill_(0)
        #
        # replace convolution by shifting: dii
        #   0  1  0
        #   0 -2  0
        #   0  1  0
        dum = batch[:, :, 0:-1, :]
        t = (zW, dum)
        down = torch.cat(t, dim=2)
        dum = batch[:, :, 1:, :]
        t = (dum, zW)
        up = torch.cat(t, dim=2)
        dii = -2 * batch + down + up
        #
        # replace convolution by shifting: di
        #   0  -0.5   0
        #   0    0    0
        #   0   0.5   0
        di = 0.5 * (up - down)
        #
        # replace convolution by shifting: djj
        #   0  0  0
        #   1 -2  1
        #   0  0  0
        dum = batch[:, :, :, 0:-1]
        t = (zH, dum)
        right = torch.cat(t, dim=3)
        dum = batch[:, :, :, 1:]
        t = (dum, zH)
        left = torch.cat(t, dim=3)
        djj = -2 * batch + right + left
        #
        # replace convolution by shifting: dj
        #      0  0    0
        #   -0.5  0  0.5
        #      0  0    0
        dj = 0.5 * (left - right)
        del left, right, zW
        #
        # replace convolution by shifting: dij
        #    1   0  -1
        #    0   0   0
        #   -1   0   1
        dum = down[:, :, :, 0:-1]
        t = (zH, dum)
        downRight = torch.cat(t, dim=3)

        dum = down[:, :, :, 1:]
        t = (dum, zH)
        downLeft = torch.cat(t, dim=3)

        dum = up[:, :, :, 0:-1]
        t = (zH, dum)
        upRight = torch.cat(t, dim=3)

        dum = up[:, :, :, 1:]
        t = (dum, zH)
        upLeft = torch.cat(t, dim=3)

        dij = 0.25 * (downRight + upLeft - downLeft - upRight)
        det = dii * djj - dij * dij
        del dum, down, up, downRight, downLeft, upRight, upLeft, zH

        #
        # DETECTION - - - - - - - - - - - -
        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        tr = dii + djj
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)
        del tr

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        #
        # LOCALIZATION - - - - - - - - - - -
        inv_hess_00 = djj / det
        inv_hess_01 = - dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        located = torch.stack([step_i, step_j], dim=1)
        return (detected, located)

# ----------------------------------------------------------------------------------------------------------------------
class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected

# ----------------------------------------------------------------------------------------------------------------------
class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        FASTMODE = True
        if (FASTMODE):
            # speedup by Rolf Lakaemper: avoid 2D convolution
            zW = torch.cuda.FloatTensor(b, c, 1, w).fill_(0)
            zH = torch.cuda.FloatTensor(b, c, h, 1).fill_(0)
            #
            # replace convolution by shifting: dii
            #   0  1  0
            #   0 -2  0
            #   0  1  0
            dum = batch[:, :, 0:-1, :]
            t = (zW, dum)
            down = torch.cat(t, dim=2)
            dum = batch[:, :, 1:, :]
            t = (dum, zW)
            up = torch.cat(t, dim=2)
            dii = -2 * batch + down + up
            #
            # replace convolution by shifting: di
            #   0  -0.5   0
            #   0    0    0
            #   0   0.5   0
            di = 0.5 * (up - down)
            #
            # replace convolution by shifting: djj
            #   0  0  0
            #   1 -2  1
            #   0  0  0
            dum = batch[:, :, :, 0:-1]
            t = (zH, dum)
            right = torch.cat(t, dim=3)
            dum = batch[:, :, :, 1:]
            t = (dum, zH)
            left = torch.cat(t, dim=3)
            djj = -2 * batch + right + left
            #
            # replace convolution by shifting: dj
            #      0  0    0
            #   -0.5  0  0.5
            #      0  0    0
            dj = 0.5 * (left - right)
            del left, right, zW
            #
            # replace convolution by shifting: dij
            #    1   0  -1
            #    0   0   0
            #   -1   0   1
            dum = down[:, :, :, 0:-1]
            t = (zH, dum)
            downRight = torch.cat(t, dim=3)

            dum = down[:, :, :, 1:]
            t = (dum, zH)
            downLeft = torch.cat(t, dim=3)

            dum = up[:, :, :, 0:-1]
            t = (zH, dum)
            upRight = torch.cat(t, dim=3)

            dum = up[:, :, :, 1:]
            t = (dum, zH)
            upLeft = torch.cat(t, dim=3)

            dij = 0.25 * (downRight + upLeft - downLeft - upRight)
            del dum, down, up, downRight, downLeft, upRight, upLeft, zH

        else:
            #This is the original (and better readable, yet slightly slower) version
            dii = F.conv2d(
                batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
            ).view(b, c, h, w)

            djj = F.conv2d(
                batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
            ).view(b, c, h, w)

            di = F.conv2d(
                batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
            ).view(b, c, h, w)

            dj = F.conv2d(
                batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
            ).view(b, c, h, w)

            dij = F.conv2d(
                batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
            ).view(b, c, h, w)

        det = dii * djj - dij * dij
        inv_hess_00 = djj / det
        inv_hess_01 = - dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        located = torch.stack([step_i, step_j], dim=1)
        return located