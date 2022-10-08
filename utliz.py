import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import metrics


def show_pointcloud(pc, size=10):
    if type(pc) == torch.Tensor:
        pc = pc.numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size)


def show_pointcloud_batch(pc, size=10):
    if type(pc) == torch.Tensor:
        pc = pc.numpy()
    if pc.shape[1]==3:
        pc = pc.transpose(0,2,1)
    B,N,C = pc.shape
    fig = plt.figure()
    for i in range(B):
        ax = fig.add_subplot(2, int(B/2), i+1, projection='3d')
        ax.scatter(pc[i, :, 0], pc[i, :, 1], pc[i, :, 2], s=size)


def show_pointcloud_2pc(pc_1, pc_2, ax=None, c1='r', c2='b',s1=1, s2=1):
    if type(pc_1) == torch.Tensor:
        pc_1 = pc_1.cpu().detach().numpy()
        pc_2 = pc_2.cpu().detach().numpy()
    if pc_1.shape[0]==3:
        pc_1 = pc_1.transpose()
        pc_2 = pc_2.transpose()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc_1[:, 0], pc_1[:, 1], pc_1[:, 2], s=s1, c=c1, alpha=0.5)
    ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 2], s=s2, c=c2, alpha=0.5)


def show_pointcloud_perpointcolor(pc, size=10,c='r'):
    # pc.shape = Nx3, c.shape = N
    if type(pc) == torch.Tensor:
        pc = pc.cpu().detach().numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    if type(c) == torch.Tensor:
        c = c.cpu().detach().numpy()
    if type(c) == np.ndarray:
        if len(c.shape) == 2:
            c = np.squeeze(c)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax0 = ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size, alpha=0.5,c=c)
    plt.colorbar(ax0, ax=ax)


def cal_auc(label, pred, pos_label=1, return_fpr_tpr=False, save_fpr_tpr=False):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=pos_label, drop_intermediate=False)
    auc_score = metrics.auc(fpr, tpr)
    if save_fpr_tpr:
        if auc_score > 0.5:
            np.save("./ROC_reinter/{:.0f}".format(auc_score * 10000),
                    np.concatenate([np.expand_dims(fpr, axis=1), np.expand_dims(tpr, axis=1)], axis=1))
    if return_fpr_tpr:
        return fpr, tpr, auc_score
    return auc_score


def cal_acc(label, pred, threshold=0.5):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    pred_logit = pred>threshold
    pred_logit = pred_logit.astype(np.long)
    acc = np.sum(pred_logit == label)/label.shape[0]
    return acc


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def cal_acc_optimThre(label, pred, pos_label=1):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=pos_label, drop_intermediate=False)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, thresholds)
    pred[pred>threshold_optimal] = 1
    pred[pred<threshold_optimal] = 0
    acc = np.sum(pred == label) / label.shape[0]
    return acc


def cal_acc_optimAccThre(label, pred, pos_label=1):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=pos_label, drop_intermediate=False)
    best_acc = 0
    for thre in thresholds:
        acc = cal_acc(label, pred, thre)
        if acc > best_acc:
            best_acc = acc
    return best_acc


def cal_TPR_TNR_FPR_FNR(label, pred):
    if type(pred) is not torch.Tensor:
        pred = torch.from_numpy(pred)
    else:
        pred = pred.detach().cpu()
    if type(label) is not torch.Tensor:
        label = torch.from_numpy(label)
    else:
        label = label.detach().cpu()

    pred_logit = pred.round()
    pseudo_label_TP = torch.sum(label * pred_logit)
    pseudo_label_TN = torch.sum((1 - label) * (1 - pred_logit))
    pesudo_label_FP = torch.sum((1 - label) * pred_logit)
    pesudo_label_FN = torch.sum(label * (1 - pred_logit))
    pseudo_label_TPR = 1.0 * pseudo_label_TP / (label.sum() + 1e-9)
    pseudo_label_TNR = 1.0 * pseudo_label_TN / (label.numel() - label.sum() + 1e-9)
    pseudo_label_FPR = 1.0 * pesudo_label_FP / (label.numel() - label.sum() + 1e-9)
    pseudo_label_FNR = 1.0 * pesudo_label_FN / (label.sum() + 1e-9)

    pseudo_label_precision = 1.0 * pseudo_label_TP / (pred_logit.sum() + 1e-9)
    pseudo_label_acc = 1.0 * torch.sum(label == pred_logit) / label.numel()
    pseudo_label_auc = cal_auc(label, pred)

    return [pseudo_label_TPR.item(), pseudo_label_TNR.item(), pseudo_label_FPR.item(), pseudo_label_FNR.item()],\
           pseudo_label_acc.item(), pseudo_label_auc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val_window = []
        self.avg_window = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if len(self.val_window)< 10:
            self.val_window.append(self.val)
        elif len(self.val_window) == 10:
            self.val_window.pop(0)
            self.val_window.append(self.val)
        else:
            print("windows avg ERROR")
        self.avg_window = np.array(self.val_window).mean()


# class VisdomLinePlotter(object):
#     """Plots to Visdom"""
#     def __init__(self, env_name='main'):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}
#         self.scatters = {}
#     def plot(self, var_name, split_name, title_name, x, y):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=title_name,
#                 xlabel='Epochs',
#                 ylabel=var_name
#             ))
#         else:
#             self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
#
#     # def scatter(self, var_name, split_name, title_name, x, size=10):
#     #     if var_name not in self.scatters:
#     #         self.scatters[var_name] = self.viz.scatter(X=x.cpu().detach().numpy(), env=self.env, opts=dict(
#     #             legend=[split_name],
#     #             title=title_name,
#     #             markersize=size
#     #         ))
#     #     else:
#     #         self.viz.scatter(X=x.cpu().detach().numpy(), env=self.env, win=self.scatters[var_name], name=split_name, update='replace')
#
#     def scatter(self, var_name, split_name, title_name, x, size=10, color=0, symbol='dot'):
#         if var_name not in self.scatters:
#             if type(x) == torch.Tensor:
#                 x = x.cpu().detach().numpy()
#             self.scatters[var_name] = self.viz.scatter(X=x, env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=title_name,
#                 markersize=size,
#                 markercolor=color,
#                 markerborderwidth=0,
#                 # opacity=0.5
#                 # markersymbol=symbol,
#                 # linecolor='white',
#             ))
#         else:
#             if type(x) == torch.Tensor:
#                 x = x.cpu().detach().numpy()
#             self.viz.scatter(X=x, env=self.env, win=self.scatters[var_name], name=split_name, update='replace')


####################################
########### plotly plot ############
def show_3D_imageSlice_plotly(volume):
    if type(volume) == torch.Tensor:
        volume = volume.detach().cpu().numpy()
    r, c = volume[0].shape
    # Define frames
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
    nb_frames = volume.shape[0]

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((nb_frames-1)/10 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-1 - k]),
        cmin=0, cmax=200
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(nb_frames-1)/10 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-1]),
        colorscale='Gray',
        cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
        ))


    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             title='Slices in volumetric data',
             width=600,
             height=600,
             scene=dict(
                        zaxis=dict(range=[-0.1, (nb_frames-1)/10], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    fig.show()


def show_3D_volume_plotly(volume, surface_count=17):
    import plotly.graph_objects as go
    import numpy as np
    import plotly.io as pio
    pio.renderers.default = "browser"
    if type(volume) == torch.Tensor:
        volume = volume.detach().cpu().numpy()

    X, Y, Z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=surface_count,  # needs to be a large number for good volume rendering
    ))
    fig.show()

####################################
####################################

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


####################################
####################################
class Network_Logger(object):
    def __init__(self, model):
        self.model = model
        self.model_grad_dict = {}
        self.model_weight_dict = {}
        self.model_weightSize_dict = {}
        for (i, j) in self.model.named_parameters():
            if len(j.shape) > 1:
                self.model_grad_dict[i] = []
                self.model_weight_dict[i] = [j.abs().mean().item()]
                self.model_weightSize_dict[i] = j.shape

    def log_grad(self):
        for (i, j) in self.model.named_parameters():
            if len(j.shape) > 1:
                self.model_grad_dict[i].append(j.grad.abs().mean().item())

    def log_weight(self):
        for (i, j) in self.model.named_parameters():
            if len(j.shape) > 1:
                self.model_weight_dict[i].append(j.abs().mean().item())

    def get_current_weight(self):
        current_weight = []
        for key in self.model_weight_dict.keys():
            current_weight.append(self.model_weight_dict[key][-1])
        return current_weight

    def get_current_grad(self):
        current_grad = []
        for key in self.model_grad_dict.keys():
            current_grad.append(self.model_grad_dict[key][-1])
        return current_grad

    def plot_grad(self, layer_idx=None):
        # example: layer_idx = [0,1,2] for only first 3 layers
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if layer_idx is not None:
            for idx, key in enumerate(self.model_grad_dict.keys()):
                if idx in layer_idx:
                    ax.plot(self.model_grad_dict[key], label=str(key))
            ax.legend()
        else:
            for idx, key in enumerate(self.model_grad_dict.keys()):
                ax.plot(self.model_grad_dict[key], label=str(key))
            ax.legend()

    def plot_weight(self, layer_idx=None):
        # example: layer_idx = [0,1,2] for only first 3 layers
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if layer_idx is not None:
            for idx, key in enumerate(self.model_weight_dict.keys()):
                if idx in layer_idx:
                    ax.plot(self.model_weight_dict[key], label=str(key))
            ax.legend()
        else:
            for idx, key in enumerate(self.model_weight_dict.keys()):
                ax.plot(self.model_weight_dict[key], label=str(key))
            ax.legend()


####################################
####################################
def show_img(img, save_file_name=''):
    if type(img) == torch.Tensor:
        img = img.cpu().detach().numpy()
    if len(img.shape) == 3:  # HxWx3 or 3xHxW, treat as RGB image
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    fig = plt.figure()
    plt.imshow(img)
    if save_file_name != '':
        plt.savefig(save_file_name, format='svg')
    plt.colorbar()
    plt.show()

def show_img_multi(img_list, num_col, num_row):
    fig = plt.figure()

    for idx, img in enumerate(img_list):
        if type(img) == torch.Tensor:
            img = img.cpu().detach().numpy()
        if len(img.shape) == 3:  # HxWx3 or 3xHxW, treat as RGB image
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
        ax = fig.add_subplot(num_col, num_row, idx+1)
        ax.imshow(img)
    plt.show()
