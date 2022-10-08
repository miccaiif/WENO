# adapted from DeepCluster repo: https://github.com/facebookresearch/deepcluster
import math
import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = [ 'AlexNet', 'alexnet_MNIST', 'alexnet', 'alexnet_STL10', 'alexnet_PCam', 'alexnet_CAMELYON',
            'AlexNet_MNIST_projection_prototype', 'alexnet_MedMNIST', 'alexnet_CIFAR10']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    'big': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'],
    'small': [(64, 11, 4, 2), 'M', (192, 5, 1, 2), 'M', (384, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1), 'M'],
    'mnist': [(32, 6, 2, 2), (64, 3, 1, 1), 'M', (128, 3, 1, 1), (128, 3, 1, 1), 'M'],
    'CAMELYON': [(96, 12, 4, 4), (256, 12, 4, 4), 'M_', (256, 5, 1, 2), 'M_', (512, 3, 1, 1), (512, 3, 1, 1), (256, 3, 1, 1), 'M_'],
    'CIFAR10': [(96, 3, 1, 1), 'M', (192, 3, 1, 1), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (192, 3, 1, 1), 'M']
}


class AlexNet(nn.Module):
    def __init__(self, features, num_classes, init=True):
        super(AlexNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(256 * 2 * 2, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True))
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(4096, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        if self.return_features: # switch only used for CIFAR-experiments
            return x
        if self.headcount == 1:
            if self.top_layer: # this way headcount can act as switch.
                x = self.top_layer(x)
            return x
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(x))
            return outp

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class AlexNet_4x4(nn.Module):
    def __init__(self, features, num_classes, init=True):
        super(AlexNet_4x4, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(256 * 4 * 4, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True))
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(4096, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        if self.return_features: # switch only used for CIFAR-experiments
            return x
        if self.headcount == 1:
            if self.top_layer: # this way headcount can act as switch.
                x = self.top_layer(x)
            return x
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(x))
            return outp

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class AlexNet_MNIST(nn.Module):
    def __init__(self, features, num_classes, init=True):
        super(AlexNet_MNIST, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(128 * 2 * 2, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.ReLU(inplace=True))
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(1024, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(1024, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 2 * 2)
        x = self.classifier(x)
        if self.return_features: # switch only used for CIFAR-experiments
            return x
        if self.headcount == 1:
            if self.top_layer: # this way headcount can act as switch.
                x = self.top_layer(x)
                # x = nn.functional.tanh(x)  # add by xiaoyuan 2021_4_22 to avoid nan in loss
            return x
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(x))
            return outp

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class AlexNet_CIFAR10(nn.Module):
    def __init__(self, features, num_classes, init=True, input_feat_dim=192*3*3):
        super(AlexNet_CIFAR10, self).__init__()
        self.features = features
        self.input_feat_dim = input_feat_dim
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(input_feat_dim, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(4096, num_classes[0])
        else:
            for a, i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x):
        if self.features is not None:
            x = self.features(x)
        x = x.view(x.size(0), self.input_feat_dim)
        x = self.classifier(x)
        if self.return_features: # switch only used for CIFAR-experiments
            return x
        if self.headcount == 1:
            if self.top_layer: # this way headcount can act as switch.
                x = self.top_layer(x)
                # x = nn.functional.tanh(x)  # add by xiaoyuan 2021_4_22 to avoid nan in loss
            return x
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(x))
            return outp

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class AlexNet_MNIST_projection_prototype(nn.Module):
    def __init__(self, output_dim=0, hidden_mlp=0, nmb_prototypes=0, init=True, normalize=True,
                 eval_mode=False, norm_layer=None):
        super(AlexNet_MNIST_projection_prototype, self).__init__()

        self.features = make_layers_features(CFG['mnist'], 1, bn=True)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            # self.projection_head = nn.Linear(128*2*2, output_dim)
            self.projection_head = nn.Linear(128, output_dim)
        else:
            self.projection_head = nn.Sequential(
                # nn.Linear(128*2*2, hidden_mlp),
                nn.Linear(128, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            # self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
            print("Multiple Prototypes is not supported now")
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_backbone(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 128 * 2 * 2)
        x = x.view(x.size(0), 128, 2 * 2)
        x = x.max(dim=-1)[0]
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        elif v == 'M_':
            layers += [nn.MaxPool2d(kernel_size=4, stride=2, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])#,bias=False)
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def alexnet(bn=True, num_classes=[1000], init=True, size='big'):
    dim = 1
    model = AlexNet(make_layers_features(CFG[size], dim, bn=bn), num_classes, init)
    return model


def alexnet_MNIST(bn=True, num_classes=[2], init=True):
    dim = 1
    model = AlexNet_MNIST(make_layers_features(CFG['mnist'], dim, bn=bn), num_classes, init)
    return model


def alexnet_MedMNIST(bn=True, num_classes=[2], init=True):
    dim = 3
    model = AlexNet_MNIST(make_layers_features(CFG['mnist'], dim, bn=bn), num_classes, init)
    return model


def alexnet_STL10(num_classes):
    model = SmallAlexNet(num_classes)
    return model


def alexnet_PCam(bn=True, num_classes=[2], init=True):
    dim = 3
    model = AlexNet(make_layers_features(CFG['big'], input_dim=dim ,bn=bn), num_classes=num_classes, init=init)
    return model


def alexnet_CAMELYON(bn=True, num_classes=[2], init=True):
    dim = 3
    model = AlexNet_4x4(make_layers_features(CFG['CAMELYON'], input_dim=dim ,bn=bn), num_classes=num_classes, init=init)
    return model


def alexnet_CIFAR10(bn=True, num_classes=[2], init=True):
    dim = 3
    model = AlexNet_CIFAR10(make_layers_features(CFG['CIFAR10'], dim, bn=bn), num_classes, init)
    return model


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=[2]):
        super(SmallAlexNet, self).__init__()
        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # fc6
        blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 7 * 7, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, num_classes[0]),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            x = layer(x)
        return x


class AlexNet_MNIST_attention(nn.Module):
    def __init__(self, features, num_classes, init=True, withoutAtten=False):
        super(AlexNet_MNIST_attention, self).__init__()
        self.withoutAtten=withoutAtten
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(128 * 2 * 2, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.ReLU(inplace=True))
        self.L = 1024
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(1024, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x, returnBeforeSoftMaxA=False):
        x = x.squeeze(0)
        x = self.features(x)
        x = x.view(x.size(0), 128 * 2 * 2)
        x = self.classifier(x)

        # Attention module
        A_ = self.attention(x)  # NxK
        A_ = torch.transpose(A_, 1, 0)  # KxN
        A = F.softmax(A_, dim=1)  # softmax over N

        if self.withoutAtten:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = torch.mm(A, x)  # KxL

        if self.return_features: # switch only used for CIFAR-experiments
            return x

        x = self.top_layer(x)
        if returnBeforeSoftMaxA:
            return x, 0, A, A_
        return x, 0, A

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class AlexNet_CIFAR10_attention(nn.Module):
    def __init__(self, features, num_classes, init=True, withoutAtten=False, input_feat_dim=192*3*3):
        super(AlexNet_CIFAR10_attention, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.withoutAtten=withoutAtten
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(input_feat_dim, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.ReLU(inplace=True))
        self.L = 1024
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(1024, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x, returnBeforeSoftMaxA=False, scores_replaceAS=None):
        if self.features is not None:
            x = x.squeeze(0)
            x = self.features(x)
        x = x.view(x.size(0), self.input_feat_dim)
        x = self.classifier(x)

        # Attention module
        A_ = self.attention(x)  # NxK
        A_ = torch.transpose(A_, 1, 0)  # KxN
        A = F.softmax(A_, dim=1)  # softmax over N

        if scores_replaceAS is not None:
            A_ = scores_replaceAS
            A = F.softmax(A_, dim=1)  # softmax over N

        if self.withoutAtten:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = torch.mm(A, x)  # KxL

        if self.return_features: # switch only used for CIFAR-experiments
            return x

        x = self.top_layer(x)
        if returnBeforeSoftMaxA:
            return x, 0, A, A_
        return x, 0, A

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class AlexNet_CIFAR10_dsmil(nn.Module):
    def __init__(self, features, num_classes, init=True, withoutAtten=False, input_feat_dim=192 * 3 * 3):
        super(AlexNet_CIFAR10_dsmil, self).__init__()
        self.withoutAtten=withoutAtten
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(input_feat_dim, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.ReLU(inplace=True))
        # self.L = 1024
        # self.D = 512
        # self.K = 1
        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
        #     nn.Linear(self.D, self.K)
        # )

        self.fc_dsmil = nn.Sequential(nn.Linear(1024, 2))
        self.q_dsmil = nn.Linear(1024, 1024)
        self.v_dsmil = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(1024, 1024)
        )
        self.fcc_dsmil = nn.Conv1d(2, 2, kernel_size=1024)

        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(1024, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x):
        if self.features is not None:
            x = x.squeeze(0)
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # # Attention module
        # A_ = self.attention(x)  # NxK
        # A_ = torch.transpose(A_, 1, 0)  # KxN
        # A = F.softmax(A_, dim=1)  # softmax over N
        #
        # if self.withoutAtten:
        #     x = torch.mean(x, dim=0, keepdim=True)
        # else:
        #     x = torch.mm(A, x)  # KxL
        #
        # if self.return_features: # switch only used for CIFAR-experiments
        #     return x
        # x = self.top_layer(x)
        # if returnBeforeSoftMaxA:
        #     return x, 0, A, A_
        # return x, 0, A

        feat = x
        device = feat.device
        instance_pred = self.fc_dsmil(feat)
        V = self.v_dsmil(feat)
        Q = self.q_dsmil(feat).view(feat.shape[0], -1)
        _, m_indices = torch.sort(instance_pred, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feat, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K
        q_max = self.q_dsmil(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc_dsmil(B) # 1 x C x 1
        C = C.view(1, -1)
        return instance_pred, C, A, B

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def alexnet_MNIST_Attention(bn=True, num_classes=[2], init=True):
    dim = 1
    model = AlexNet_MNIST_attention(make_layers_features(CFG['mnist'], dim, bn=bn), num_classes, init)
    return model


def alexnet_CIFAR10_Attention(bn=True, num_classes=[2], init=True):
    dim = 3
    model = AlexNet_CIFAR10_attention(make_layers_features(CFG['CIFAR10'], dim, bn=bn), num_classes, init)
    return model


########################################
## models for Shared Stu and Tea network
def alexnet_CIFAR10_Encoder():
    dim = 3
    model = make_layers_features(CFG['CIFAR10'], dim, bn=True)
    return model


def teacher_Attention_head(bn=True, num_classes=[2], init=True, input_feat_dim=192*3*3):
    model = AlexNet_CIFAR10_attention(features=None, num_classes=num_classes, init=init, input_feat_dim=input_feat_dim)
    return model


def teacher_DSMIL_head(bn=True, num_classes=[2], init=True, input_feat_dim=192*3*3):
    model = AlexNet_CIFAR10_dsmil(features=None, num_classes=num_classes, init=init, input_feat_dim=input_feat_dim)
    return model


def student_head(num_classes=[2], init=True, input_feat_dim=192*3*3):
    model = AlexNet_CIFAR10(None, num_classes, init, input_feat_dim=input_feat_dim)
    return model


class feat_projecter(nn.Module):
    def __init__(self, input_feat_dim=512, output_feat_dim=512):
        super(feat_projecter, self).__init__()
        # self.projecter = nn.Sequential(
        #     nn.Linear(input_feat_dim, input_feat_dim*2),
        #     nn.BatchNorm1d(input_feat_dim*2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_feat_dim*2, input_feat_dim * 2),
        #     nn.BatchNorm1d(input_feat_dim*2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_feat_dim * 2, output_feat_dim),
        #     nn.BatchNorm1d(output_feat_dim),
        # )
        self.projecter = nn.Sequential(
            nn.Linear(input_feat_dim, output_feat_dim),
            nn.BatchNorm1d(output_feat_dim)
        )
    def forward(self, x):
        x = self.projecter(x)
        return x


def camelyon_feat_projecter(input_dim, output_dim):
    model = feat_projecter(input_dim, output_dim)
    return model
########################################

if __name__ == '__main__':
    import torch
    # model = alexnet(num_classes=[500]*3)
    # print([ k.shape for k in model(torch.randn(64,3,224,224))])
    model = AlexNet_MNIST_projection_prototype(output_dim=128, hidden_mlp=2048, nmb_prototypes=300)
    print("END")

