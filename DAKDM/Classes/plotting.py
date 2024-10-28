import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.animation
import math
from scipy.interpolate import LinearNDInterpolator
from Classes.colors import parula_color
from Classes.utils import mkdir

def asCartesian(r, theta, phi, M = 8, N = 8, Ng = 352):
    # M, N antennas. Ng cyclic prefix duration

    #takes list rthetaphi (single coord)
    theta   = 90/8 * theta* np.pi/180 # to radian
    phi     = 360/8 * phi* np.pi/180
    # r = 100/Ng * r
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]

def asCartesian2D(r, theta_phi, M = 64, Ng = 352):
    # M, N antennas. Ng cyclic prefix duration

    #takes list rthetaphi (single coord)
    theta_phi   = 180/64 * theta_phi* np.pi/180 # to radian
    # r = 100/Ng * r
    x = r * np.cos( theta_phi )
    y = r * np.sin( theta_phi )
    return [x,y]  

def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def toDeciBellMilliWatt(RSSI):
    """Returns RSSI value in dBm, assuming input is mW"""
    if RSSI <= 0:
        RSSI = np.nan
    return 10 * math.log10(RSSI)

def format_dBm(value, tick_number):
    if value <= 0:
        return ''
    return '-' + str(round(-toDeciBellMilliWatt(value)))


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



def plot_likelihood(train_score, test_score, test_labels, log_path, file_name, prove = 1, log_scale = True, bins=100, threshold = None, remove_outliers = None, xlim = None, save_eps = 0, ax = None, save_pdf = 1, plt_show = 1, var_to_monitor = 'energy', only_normal_data = 0, relim = 1):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    energy_normals_train = train_score.ravel()
    energy_normals_test = test_score[np.squeeze(test_labels == 0)].ravel()
    energy_anomalies_test = test_score[np.squeeze(test_labels == 1)].ravel()
    np.unique(energy_anomalies_test)

    if remove_outliers is not None:
        energy_normals_train = energy_normals_train[~is_outlier(energy_normals_train)]
        energy_normals_test = energy_normals_test[~is_outlier(energy_normals_test)]
        energy_anomalies_test = energy_anomalies_test[~is_outlier(energy_anomalies_test)]

    axis_ = []
    plt.sca(ax)  # Use the pyplot interface to change just one subplot

    if prove:
        sns.histplot(energy_normals_train, bins = bins, log_scale=log_scale, color = 'blue')
        sns.histplot(energy_normals_test, bins = bins, log_scale=log_scale,color = 'tab:blue')
        if not only_normal_data:
            sns.histplot(energy_anomalies_test, bins = bins, log_scale=log_scale, color = 'tab:red')
        pass
    else:
        sns.kdeplot(energy_normals_train, bw_adjust=1, cut= 3, linewidth=2, label=f'training', color = 'blue', linestyle='--')
        if relim:
            ax.relim() # the axis limits need to be recalculated without the bars
            ax.autoscale_view()

        sns.kdeplot(energy_normals_test, bw_adjust=1, cut= 3, linewidth=2, label=f'validation', color = 'tab:blue')
        if relim:
            ax.relim() # the axis limits need to be recalculated without the bars
            ax.autoscale_view()

        if not only_normal_data: 

            sns.kdeplot(energy_anomalies_test, bw_adjust=1, cut= 3, linewidth=2, label=f'validation', color = 'tab:red')
            if relim:
                ax.relim() # the axis limits need to be recalculated without the bars
                ax.autoscale_view()

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    plt.ylabel('Density')            
    if var_to_monitor == 'likelihood1' or var_to_monitor == 'likelihood2':
        plt.xlabel('Likelihood')
    elif var_to_monitor == 'energy1' or var_to_monitor == 'energy2':   
        plt.xlabel('Energy')
    # ax.set(xlabel='', ylabel='')
    # ax.title.set_text('')
    ax.grid()

    # plot threshold
    if threshold is not None:
        plt.axvline(threshold, color='black')
        pass

    if threshold is not None:
        lines = [Line2D([0], [0], color='blue', linewidth=2, linestyle='--'), 
                Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-'),
                Line2D([0], [0], color='tab:red', linewidth=2, linestyle='-'),
                Line2D([0], [0], color='black', linewidth=2, linestyle='-')]
        labels = ['normal training data', 'normal validation data', 'anomaly validation data', 'threshold',]

    else:
        if not only_normal_data: 
            lines = [Line2D([0], [0], color='blue', linewidth=2, linestyle='--'), 
                    Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-'),
                    Line2D([0], [0], color='tab:red', linewidth=2, linestyle='-')]
            labels = ['normal training data', 'normal validation data', 'anomaly validation data',] 
        else:
            lines = [Line2D([0], [0], color='blue', linewidth=2, linestyle='--'), 
                    Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-')]
            labels = ['normal training data', 'normal validation data'] 
    plt.legend(lines, labels)

    if save_pdf:
        plt.savefig(log_path + f'/{file_name}.pdf', bbox_inches='tight')
    if save_eps:
        plt.savefig(log_path + f'/{file_name}.eps', format='eps', bbox_inches='tight')
    if plt_show:
        plt.show()


def plot_many_likelihood(scores, names, log_path, file_name, prove = 1, log_scale = True, logy = 0, bins=100, threshold = None, remove_outliers = None, xlim = None, ylim = None, save_eps = 0, ax = None, save_pdf = 1, plt_show = 1, var_to_monitor = 'energy2'):

    # Desired figure and axes size (in inches)
    fig_width_inches = 6.4
    fig_height_inches = 4.8
    axes_width_inches = 4.96
    axes_height_inches = 3.5


    if ax is None:
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches)) # default size
        ax = fig.add_subplot(111)
        plt.sca(ax)  # Use the pyplot interface to change just one subplot

    for name in names:
        i = names.index(name)
        energy = scores[i].ravel()
        np.unique(energy)

        if remove_outliers is not None:
            energy = energy[~is_outlier(energy)]

        axis_ = []

        if prove:
            sns.histplot(energy, bins = bins, log_scale=log_scale, label=name)
            pass
        else:
            # sns.histplot(energy, kde=True, stat='density', binwidth=1, log_scale=log_scale, kde_kws={'cut': 3}, line_kws = {'linewidth':2}, label=name)
            # ax.containers[0].remove() # remove the bars

            sns.kdeplot(energy, bw_adjust=1, cut= 3, linewidth=2, log_scale=log_scale)
            ax.relim() # the axis limits need to be recalculated without the bars
            ax.autoscale_view()

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if logy:
        plt.yscale('log', nonposy='clip')
    plt.ylabel('Density')            
    if var_to_monitor == 'likelihood1' or var_to_monitor == 'likelihood2':
        plt.xlabel('Likelihood')
    elif var_to_monitor == 'energy1' or var_to_monitor == 'energy2':   
        plt.xlabel('Energy')
    # ax.set(xlabel='', ylabel='')
    # ax.title.set_text('')
    ax.grid()

    # plot threshold
    if threshold is not None:
        plt.axvline(threshold, color='black')
        pass

    if threshold is not None:
        plt.legend(names)
    else:
        plt.legend(names + ['Threshold'])
    
    # adjust axes size
    # Initialize the loop variables
    current_axes_width = 0
    current_axes_height = 0
    delta = 0.001
    left, right, bottom, top = 0, 1, 0, 1
    # Loop until the desired axes size is met
    while abs(current_axes_width - axes_width_inches) > 0.01 or abs(current_axes_height - axes_height_inches) > 0.01:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        
        ax = plt.gca()
        ax_position = ax.get_position()
        
        current_axes_width = fig_width_inches * (ax_position.x1 - ax_position.x0)
        current_axes_height = fig_height_inches * (ax_position.y1 - ax_position.y0)
        
        if current_axes_width > axes_width_inches:
            right -= delta
        elif current_axes_width < axes_width_inches:
            right += delta
            
        if current_axes_height > axes_height_inches:
            top -= delta
        elif current_axes_height < axes_height_inches:
            top += delta

    if save_pdf:
        plt.savefig(log_path + f'/{file_name}.pdf', bbox_inches='tight')
    if save_eps:
        plt.savefig(log_path + f'/{file_name}.eps', format='eps', bbox_inches='tight')
    if plt_show:
        plt.show()


def plot_metrics_varying_thresholds(thrs, acc, recall, precision, f_score, log_path, file_name = 'thresholds'):

    plt.plot(thrs, acc)
    plt.plot(thrs, recall)
    plt.plot(thrs, precision)
    plt.plot(thrs, f_score)
    plt.grid()
    plt.legend(['accuracy', 'recall', 'precision', 'f_score'])
    plt.xlabel('Threshold')
    plt.savefig(log_path + f'/{file_name}s.pdf', bbox_inches='tight')
    plt.savefig(log_path + f'/{file_name}.eps', format='eps', bbox_inches='tight')

    # plt.ylim(0.98,1);
    idx = np.argwhere(np.diff(np.sign(np.array(precision) - np.array(recall)))).flatten()
    print(f'Threshold where FP = FN: {thrs[idx]}')
    acc[idx[0]]
    np.max(acc), np.argmax(acc)


def plot_roc_auc_curve(fpr, tpr, auc, title='ROC Curve', file_name = 'ROC_curve' ,dir = None, save_eps = 0):
    plt.plot([0, 1], [0, 1], color='lightgray', lw=2, linestyle='--')
    plt.plot(fpr, tpr, lw=2)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' - AUC = ' + str(auc))
    plt.tight_layout()
    if dir is not None:
        plt.savefig(dir + f'/{file_name}.pdf', bbox_inches='tight')
        if save_eps:
            plt.savefig(dir + f'/{file_name}.eps', format='eps', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_auc_gmm_components(roc_aucs, components, title='AUC vs GMM components',  file_name = 'AUC_vs_GMM', dir = None):
    # plt.rcParams.update({'font.size': 18})
    plt.xticks(fontsize=10, rotation=0)
    plt.plot(components, roc_aucs, lw=2)
    # plt.xlim([-0.01, 1.01])
    plt.xticks(components)
    # plt.ylim([-0.01, 1.01])
    plt.xlabel('N. of clusters')
    plt.ylabel('AUC')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.xticks(np.arange(0, len(components) + 1, 2)) 
    if dir is not None:
        plt.savefig(dir + f'/{file_name}.pdf', bbox_inches='tight')
        plt.savefig(dir + f'/{file_name}.eps', format='eps', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    max_auc = max(roc_aucs)
    opt_gmm_components = components[roc_aucs.index(max_auc)]
    print(f'Max AUC: {max_auc}, Optimum gmm components: {opt_gmm_components} ')
    # plt.rcParams.update({'font.size': 20})



def plot_reconstructed_samples (input_, sample_number, label, log_path, file_name = '', sample_energy = '', output_ = None, save_pdf = 1, save_png = 1, save_eps = 0, plt_show = 0, fontsize = 20, remove_axis_ticks = 0, flat_image = 0, polar = 0, cmap='parula'):

    if cmap == 'parula':
        cmap = parula_color()
    else:
        cmap = 'jet'

    if not polar:
            
        X = np.arange(0, input_.shape[1], 1)
        Y = np.arange(0, input_.shape[0], 1)
        X, Y = np.meshgrid(X, Y)
        #X, Y = np.mgrid[0:input_.shape[0], 0:input_.shape[1]]
        X.shape
        Y.shape
        fig = plt.figure(figsize=(20,20))
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')

        # POSITIONING PAPER for no background but keeping axes
        # Make the grid and labels opaque
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        # Make the panes (background) transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Make the figure background transparent
        fig.patch.set_alpha(0)

        surf = ax.plot_surface(X, Y, input_, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)

        ax.grid(False)

        if remove_axis_ticks:
            plt.axis('off')
        else:
            plt.minorticks_off()
            ax.zaxis.set_major_formatter(plt.FuncFormatter(format_dBm))
            plt.xticks(fontsize=fontsize, rotation=0)
            plt.yticks(fontsize=fontsize, rotation=0)
            ax.zaxis.set_tick_params(labelsize=fontsize)
            ax.invert_xaxis()

        if flat_image:
            # Flat image
            plt.show(block=False)
            plt.pause(1)
            vmin, vmax =  surf.get_clim()
            plt.pause(1)
            surf.set_clim(vmin, vmax/3) 
            ax.view_init(90, 0)
        else:


            # adjust colormap (POSITIONING PAPER)
            plt.show(block=False)
            plt.pause(1)
            vmin, vmax =  surf.get_clim()
            plt.pause(1)
            surf.set_clim(vmin, vmax/3) 

            # vertical for input image (POSITIONING PAPER)
            # ax.view_init(13, -117)
            ax.view_init(30, -40)

    else:


        x_dim = int(input_.shape[0]) # 352
        y_dim = int(input_.shape[1]) # 64

        x = np.arange(0, x_dim, 1)
        y = np.arange(0, y_dim, 1)
        X,Y = np.meshgrid(y, x) # X (352, 64), Y (352, 64)
        interp = LinearNDInterpolator(np.stack((X.flatten(), Y.flatten()), 1), input_.flatten())

        max_ind = np.argmax(input_, 0)[0]       # index of delays with highest power
        max_phi_theta = np.argmax(input_, 1)[0] # index of angles with highest power
        max_ = np.max(input_.flatten())


        # interpolate angles
        y = np.arange(0, y_dim, 0.1)
        X,Y = np.meshgrid(y, x) # X (352, 640), Y (352, 640)
        input_ = interp(X, Y)


        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')

        X_cart, Y_cart = asCartesian2D(Y, X)

        surf = ax.plot_surface(X_cart, Y_cart, input_, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)
        
        plt.show(block=False)
        plt.pause(1)
        vmin, vmax =  surf.get_clim()
        plt.pause(1)
        surf.set_clim(vmin, vmax/2) 
        if remove_axis_ticks:
            plt.axis('off')
        
        ax.view_init(30, -40)


    if save_pdf:
        plt.savefig(log_path + f'/{file_name}_input_sample_{sample_number}_label_{label}_energy_{sample_energy}_flat_{flat_image}_polar_{polar}.pdf', bbox_inches='tight')
    if save_png:
        plt.savefig(log_path + f'/{file_name}_input_sample_{sample_number}_label_{label}_energy_{sample_energy}_flat_{flat_image}_polar_{polar}.png', bbox_inches='tight')
    if save_eps:
        plt.savefig(log_path + f'/{file_name}_input_sample_{sample_number}_label_{label}_energy_{sample_energy}_flat_{flat_image}_polar_{polar}.eps', format='eps', bbox_inches='tight')
    if plt_show:
        plt.show()

    if output_ is not None:
        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, output_, rstride=1, cstride=1, cmap=parula_color(), linewidth=0, antialiased=False)
        #ax.set_zlim(-1.01, 1.01)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xticks(fontsize=fontsize, rotation=0)
        plt.yticks(fontsize=fontsize, rotation=0)
        ax.zaxis.set_tick_params(labelsize=fontsize)
        ax.view_init(15, -30)
        plt.savefig(log_path + f'/{file_name}_output_sample_{sample_number}_label_{label}_energy_{sample_energy}.pdf', bbox_inches='tight')
        if save_eps:
            plt.savefig(log_path + f'/{file_name}_output_sample_{sample_number}_label_{label}_energy_{sample_energy}.eps', format='eps', bbox_inches='tight')
        if plt_show:
            plt.show()
    plt.clf()
    plt.close()


def plot_sparse_reconstructed_samples (input_, sample_number, label, log_path, file_name = '', sample_energy = '', output_ = None, save_eps = 0, plt_show = 0, fontsize = 20, polar = 0):

    try:

        x_dim = int(np.sqrt(input_.shape[1]))
        y_dim = x_dim
        z_dim = input_.shape[0]

        input_ = input_.reshape((z_dim, x_dim, y_dim))
        x = np.arange(0, x_dim, 1)
        y = np.arange(0, y_dim, 1)
        z = np.arange(0, z_dim, 1)
        X,Y,Z = np.meshgrid(y, z, x)
        interp = LinearNDInterpolator(np.stack((X.flatten(), Y.flatten(), Z.flatten()), 1), input_.flatten())

        # x = np.r_[np.arange(0, 4, 0.25), np.arange(4, 8, 0.5)]
        # y = np.r_[np.arange(0, 4, 0.5), np.arange(4, 8, 0.25)]
        # z = np.r_[np.arange(55, 57, 0.5), np.arange(57, 60, 0.25), np.arange(60, 65, 0.25)]

        max_ind = np.argmax(input_, 0)[0][0]
        max_phi = np.argmax(input_, 1)[0][0]
        max_theta = np.argmax(input_, 2)[0][0]
        max_ = np.max(input_.flatten())
        if polar == 0:

            x = np.arange(0, x_dim, 0.1) #+ np.random.normal(0,1, int(x_dim*1/0.25))
            y = np.arange(0, y_dim, 0.1) #+ np.random.normal(0,1, int(y_dim*1/0.25))
            z = np.arange(max_ind-7, max_ind+7, 0.1) #+ np.random.normal(0,1, int((65-55)*1/0.25))
            X,Y,Z = np.meshgrid(y, z, x)
            input_ = interp(X, Y, Z)
            input_[input_<7] = np.nan
            X = X[~np.isnan(input_)]
            Y = Y[~np.isnan(input_)]
            Z = Z[~np.isnan(input_)]
            input_ = input_[~np.isnan(input_)]

            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')

            p = ax.scatter(X, Y, Z, c=input_, 
                        marker='o', cmap=parula_color(), 
                        alpha = 1, #(input_-np.min(input_))/(np.max(input_)-np.min(input_)), 
                        vmin=0, vmax=max_)
            p._facecolors[:, 3] = np.clip((input_-np.min(input_))/(np.max(input_)-np.min(input_)) + 0.1, 0, 1)  # 0.2
            c = p._facecolors

            ax.clear()

            # cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
            cbar = fig.colorbar(p, fraction=0.046, pad=0.1)
            
            p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=c, 
                        marker='o', cmap=parula_color(), 
                        vmin=0, vmax=max_,
                        edgecolors='none')
            plt.ylim([max_ind-7, max_ind+7]) # [55, 65]


            h, yedges, zedges = np.histogram2d(Y.flatten(), Z.flatten(), bins=50)
            h = h.transpose()
            normalized_map = plt.cm.Blues(h/h.max())
            yy, zz = np.meshgrid(yedges, zedges)
            xpos = ax.get_xlim()[0]-0 # Plane of histogram
            xflat = np.full_like(yy, xpos) 
            p = ax.plot_surface(xflat, yy, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

            h, xedges, zedges = np.histogram2d(X.flatten(), Z.flatten(), bins=50)
            h = h.transpose()
            normalized_map = plt.cm.Blues(h/h.max())
            xx, zz = np.meshgrid(xedges, zedges)
            ypos = ax.get_ylim()[1]+0 # Plane of histogram
            yflat = np.full_like(xx, ypos) 
            p = ax.plot_surface(xx, yflat, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

            h, xedges, yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=50)
            h = h.transpose()
            normalized_map = plt.cm.Blues(h/h.max())
            xx, yy = np.meshgrid(xedges, yedges)
            zpos = ax.get_zlim()[0]-0 # Plane of histogram
            zflat = np.full_like(xx, zpos) 
            p = ax.plot_surface(xx, yy, zflat, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

            plt.minorticks_off()
            ax.zaxis.set_major_formatter(plt.FuncFormatter(format_dBm))
            plt.xticks(fontsize=fontsize, rotation=0)
            plt.yticks(fontsize=fontsize, rotation=0)

            ax.zaxis.set_tick_params(labelsize=fontsize)

            a=ax.get_xticks().tolist()
            a = [str(int(float(el)*360/x_dim ))  for el in a]
            ax.set_xticklabels(a)
            ax.set_xlim([0.1, 6.9])

            a=ax.get_yticks().tolist()
            a = [str(int(float(el)*(8.6)))  for el in a]
            ax.set_yticklabels(a)
            ax.set_ylim([52.1, 65.9])

            a=ax.get_zticks().tolist()
            a = [str(int(float(el)*180/y_dim ))  for el in a]
            ax.set_zticklabels(a, ha='left' )#va='bottom', )
            ax.set_zlim([-0.9, 7.9])

            ax.view_init(27, -53)
           
            plt.savefig(log_path + f'/{file_name}_sparse_input_sample_{sample_number}_label_{label}_energy_{sample_energy}.pdf', bbox_inches='tight')
            if save_eps:
                plt.savefig(log_path + f'/{file_name}_sparse_input_sample_{sample_number}_label_{label}_energy_{sample_energy}.eps', format='eps', bbox_inches='tight')
            if plt_show:
                plt.show()

        else:

            x = np.arange(max(0, max_theta-2), min(x_dim, max_theta+2), 0.1) # np.arange(0, x_dim, 0.1)
            y = np.arange(max(0, max_phi-2), min(y_dim, max_phi+2), 0.1) # np.arange(0, y_dim, 0.1) 
            z = np.arange(0, min(z_dim, max_ind), 0.1) #+ np.random.normal(0,1, int((65-55)*1/0.25))
            X,Y,Z = np.meshgrid(y, z, x)
            input_ = interp(X, Y, Z)
            input_[input_<0] = np.nan
            X = X[~np.isnan(input_)]
            Y = Y[~np.isnan(input_)]
            Z = Z[~np.isnan(input_)]
            input_ = input_[~np.isnan(input_)]

            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')

            X_cart, Y_cart, Z_cart = asCartesian(Y, Z, X)

            p = ax.scatter(X_cart, Y_cart, Z_cart, c=input_, 
                        marker='o', cmap=parula_color(), 
                        alpha = 1, #(input_-np.min(input_))/(np.max(input_)-np.min(input_)), 
                        vmin=0, vmax=max_)
            p._facecolors[:, 3] = np.clip((input_-np.min(input_))/(np.max(input_)-np.min(input_)) - 0.1, 0, 1)  # 0.2
            c = p._facecolors

            ax.clear()

            cbar = fig.colorbar(p)
            p = ax.scatter(X_cart.flatten(), Y_cart.flatten(), Z_cart.flatten(), c=c, 
                        marker='o', cmap=parula_color(), 
                        vmin=0, vmax=max_,
                        edgecolors='none')

            # azimuth
            R = np.linspace(0, max_ind, 100)
            h = 0
            u_pi = np.linspace(0,  np.pi, 100)
            u_2pi = np.linspace(0,  2*np.pi, 100)
            x_ = np.outer(R, np.cos(u_2pi))
            y_ = np.outer(R, np.sin(u_2pi))
            zflat = np.full_like(x_, h) 
            ax.plot_surface(x_,-y_,zflat, rstride=1, cstride=1, shade=False, color='tab:blue', alpha=0.5, linewidth=0)
            ax.plot(max_ind*np.cos(u_2pi), max_ind*np.sin(u_2pi), h, color='tab:blue')

            # ax.set_xticks(locs)
            # ax.set_xticklabels(labels)

            # elevation
            x_ = np.outer(R, np.cos(u_pi))
            y_ = np.outer(R, np.sin(u_pi))
            zflat = np.full_like(x_, h) 
            yflat = zflat
            h = np.full_like(max_ind*np.cos(u_pi), 0) 
            z_ = y_
            ax.plot_surface(x_,yflat,z_, rstride=1, cstride=1, shade=False, color='tab:blue', alpha=0.5, linewidth=0)
            ax.plot(max_ind*np.cos(u_pi), h, max_ind*np.sin(u_pi), color='tab:blue')


            ax.set_zlim(0, max_ind)

            plt.xticks(fontsize=fontsize, rotation=0)
            plt.yticks(fontsize=fontsize, rotation=0)
            ax.zaxis.set_tick_params(labelsize=fontsize)
            plt.axis('off')
            ax.view_init(27, -53)        
            plt.savefig(log_path + f'/{file_name}_sparse_polar_input_sample_{sample_number}_label_{label}_energy_{sample_energy}.pdf', bbox_inches='tight')
            if save_eps:
                plt.savefig(log_path + f'/{file_name}_sparse_polar_input_sample_{sample_number}_label_{label}_energy_{sample_energy}.eps', format='eps', bbox_inches='tight')
            if plt_show:
                plt.show()
            
    except:
        pass


def plot_latent_features(train_z, test_z, test_labels, log_path, file_name, prove = 1, symlogx_logx_logy_scale=[True, False, True], bins=100, fill = True, threshold = None, xlim = None, save_eps = 0):

    fig = plt.figure(figsize=(24,10));
    fig.tight_layout();

    symlogx, logx, logy = symlogx_logx_logy_scale
    n_samples, n_features = test_z.shape
    n_samples, n_features
    z_normals = test_z[np.squeeze(test_labels == 0),:]
    z_anomalies = test_z[np.squeeze(test_labels == 1),:]


    for feature in range(n_features):
        ax = fig.add_subplot(1, n_features, feature+1);

        axis_ = []
        plt.sca(ax);  # Use the pyplot interface to change just one subplot
        
        if prove:
            sns.histplot(train_z[:,feature], bins = bins, log_scale=logx, color = 'blue', fill=fill, element="step")
            sns.histplot(z_normals[:,feature], bins = bins, log_scale=logx, color = 'tab:blue', fill=fill, element="step")
            sns.histplot(z_anomalies[:,feature], bins = bins, log_scale=logx, color='tab:red', fill=fill)
            # sns.histplot([train_z[:,feature], z_normals[:,feature], z_anomalies[:,feature]], log_scale=logx, color = ['blue', 'tab:blue', 'tab:red'])
        else:
            sns.histplot(train_z[:,feature], kde=True, stat='density', log_scale=logx, binwidth=1, kde_kws={'cut': 3}, line_kws = {'linewidth':2}, label=f'training', color = 'blue', linestyle='--')
            # axis_.append(temp)
            ax.containers[0].remove(); # remove the bars
            ax.relim(); # the axis limits need to be recalculated without the bars
            ax.autoscale_view();
            
            sns.histplot(z_normals[:,feature], kde=True, stat='density', log_scale=logx, binwidth=1, kde_kws={'cut': 3}, line_kws = {'linewidth':2}, label=f'training', color = 'tab:blue')
            # axis_.append(temp)
            ax.containers[0].remove(); # remove the bars
            ax.relim(); # the axis limits need to be recalculated without the bars
            ax.autoscale_view();
            
            sns.histplot(z_anomalies[:,feature], kde=True, stat='density', log_scale=logx, binwidth=1, kde_kws={'cut': 3}, line_kws = {'linewidth':2}, label=f'training', color = 'tab:red')
            # axis_.append(temp)
            ax.containers[0].remove(); # remove the bars
            ax.relim(); # the axis limits need to be recalculated without the bars
            ax.autoscale_view();

        plt.xticks(rotation=0, ha='right');
        plt.subplots_adjust(bottom=0.30);
        if xlim is not None:
            plt.xlim(xlim)
        plt.ylabel('');
        plt.title('Density');
        plt.xlabel(fr'$\alpha_{{{feature}}}$', fontsize=20);
        # ax.set(xlabel='', ylabel='')
        # ax.title.set_text('')
        ax.grid();
        if logy:
            plt.yscale('log', nonposy='clip')
        if symlogx:
            plt.xscale('symlog')

    lines = [Line2D([0], [0], color='blue', linewidth=2, linestyle='--'), 
            Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-'),
            Line2D([0], [0], color='tab:red', linewidth=2, linestyle='-'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='-')];
    labels = ['normal training data', 'normal validation data', 'anomaly validation data',];
    plt.legend(lines, labels);

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
    plt.savefig(log_path + f'/{file_name}.pdf', bbox_inches='tight')
    if save_eps:
        plt.savefig(log_path + f'/{file_name}.eps', format='eps', bbox_inches='tight')
    plt.show()


def plot_training_metrics (config, log_path, batches_per_epoch, file_name='', save_eps = 0, save_pdf = 1, plt_show = 1):

    loss_logs = np.load(log_path + '/log_loss.npy', allow_pickle=True).tolist()

    epochs = np.arange(0, config['num_epochs'], 1)
    epochs = epochs * batches_per_epoch / config['log_step']

    for k,v in loss_logs.items():

        fig = plt.figure(figsize=(10, 10))
        plt.plot(v)
        [plt.axvline(e, color='black', alpha=0.5) for e in epochs]
        # plt.ylim([0, 100])
        plt.xlim([0, len(v)])
        plt.title(k.replace('_',' '))
        plt.xlabel('log step')
        if save_pdf:
            plt.savefig(log_path + f'/{file_name}_{k}.pdf', bbox_inches='tight')
        if save_eps:
            plt.savefig(log_path + f'/{file_name}_{k}.eps', format='eps', bbox_inches='tight')
        if plt_show:
            plt.show() 



def energy_per_epoch(config, epochs_to_test, train_loader, val_loader, solver_class, log_path, model_save_path, model_save_step = 586, fpr = 0.2, prove = 1, log_scale = True, bins=100, threshold = None, remove_outliers = None, xlim = None, file_name='', save_eps = 0, plt_show = 1, only_normal_data = 0, relim = 1):
    
    dir_energy = log_path + '/energy_per_epoch'
    mkdir(dir_energy)
    
    for e in epochs_to_test:
        
        config['pretrained_model'] = f'{e}_{model_save_step}' 

        solver = solver_class(config)  
        
        # Training energy computation
        solver.compute_threshold(train_loader, fpr=fpr)
        
        # Validation energy computation
        accuracy, precision, recall, f_score = solver.test(val_loader)
        
        # Save results
        np.save(dir_energy + f'/{file_name}_epoch_{e}.npy', {'train_z1': solver.train_z1, 
                                                'train_z2': solver.train_z2, 
                                                'train_likelihood1': solver.train_likelihood1, 
                                                'train_energy1': solver.train_energy1, 
                                                'train_energy2':solver.train_energy2, 
                                                'test_z1': solver.test_z1, 
                                                'test_z2': solver.test_z2, 
                                                'test_likelihood1': solver.test_likelihood1,
                                                'test_energy1': solver.test_energy1,
                                                'test_energy2': solver.test_energy2,
                                                'test_labels': solver.test_labels}, 
                                                allow_pickle = True)
    
        # Plot    
        try:
            if config['var_to_monitor'] == 'likelihood1':
                plot_likelihood(train_score=solver.train_likelihood1.ravel(), test_score=solver.test_likelihood1.ravel(), test_labels=solver.test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, threshold = threshold, xlim = xlim, save_eps = save_eps, var_to_monitor='likelihood1', 
                            plt_show = plt_show, only_normal_data = only_normal_data, relim = relim)
                            
            elif config['var_to_monitor'] == 'energy1':
                plot_likelihood(train_score=solver.train_energy1.ravel(), test_score=solver.test_energy1.ravel(), test_labels=solver.test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, threshold = threshold, xlim = xlim, save_eps = save_eps, var_to_monitor='energy1',
                            plt_show = plt_show, only_normal_data = only_normal_data, relim = relim)

            elif config['var_to_monitor'] == 'likelihood2':
                plot_likelihood(train_score=solver.train_likelihood2.ravel(), test_score=solver.test_likelihood2.ravel(), test_labels=solver.test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, threshold = threshold, xlim = xlim, save_eps = save_eps, var_to_monitor='likelihood2',
                            plt_show = plt_show, only_normal_data = only_normal_data, relim = relim)

            elif config['var_to_monitor'] == 'energy2':
                plot_likelihood(train_score=solver.train_energy2.ravel(), test_score=solver.test_energy2.ravel(), test_labels=solver.test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, threshold = threshold, xlim = xlim, save_eps = save_eps, var_to_monitor='energy2',
                            plt_show = plt_show, only_normal_data = only_normal_data, relim = relim)
        except:
            pass


def energy_moving(config, epochs_to_test, log_path, prove = 1, log_scale = True, bins=100, file_name=''):
        
    dir_energy = log_path + '/energy_per_epoch'

    fig, ax1 = plt.subplots(nrows=1)
    def update(i,epochs_to_test):

        e = epochs_to_test[i]

        # Load results
        ris = np.load(dir_energy + f'/{file_name}_epoch_{e}.npy', allow_pickle = True).tolist()

        train_z1= ris['train_z1'] 
        train_z2= ris['train_z2']
        train_likelihood1= ris['train_likelihood1'] 
        train_energy1= ris['train_energy1']
        train_energy2=ris['train_energy2'] 
        test_z1= ris['test_z1']
        test_z2= ris['test_z2']
        test_likelihood1= ris['test_likelihood1']
        test_energy1= ris['test_energy1']
        test_energy2= ris['test_energy2']
        test_labels= ris['test_labels']

        try:

            if config['var_to_monitor'] == 'likelihood1':
                plot_likelihood(train_score=train_likelihood1.ravel(), test_score=test_likelihood1.ravel(), test_labels=test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, ax = ax1, save_pdf = 0, plt_show = 0, var_to_monitor='likelihood1')
                            
            elif config['var_to_monitor'] == 'energy1':
                plot_likelihood(train_score=train_energy1.ravel(), test_score=test_energy1.ravel(), test_labels=test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, ax = ax1, save_pdf = 0, plt_show = 0, var_to_monitor='energy1')

            elif config['var_to_monitor'] == 'likelihood2':
                plot_likelihood(train_score=train_likelihood2.ravel(), test_score=test_likelihood2.ravel(), test_labels=test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, ax = ax1, save_pdf = 0, plt_show = 0, var_to_monitor='likelihood2')
                            
            elif config['var_to_monitor'] == 'energy2':
                plot_likelihood(train_score=train_energy2.ravel(), test_score=test_energy2.ravel(), test_labels=test_labels.ravel(), 
                            log_path=dir_energy, file_name=f'/{file_name}_epoch_{e}', prove = prove,
                            log_scale = log_scale, bins=bins, ax = ax1, save_pdf = 0, plt_show = 0, var_to_monitor='energy2')
        except:
            pass
        # print(i)
        if i == len(epochs_to_test)-1:
            plt.cla()


    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(epochs_to_test), repeat=True, fargs=(epochs_to_test,), interval = 1000)  
    plt.show()

    

def plot_roc_auc_acc(config, roc_aucs, fpr, acc, recall, precision, fscore, epochs, sw = None, title='AUC vs Epochs',  file_name = 'AUC_vs_epochs', dir = None):
    
    # plt.rcParams.update({'font.size': 18})
    plt.xticks(fontsize=10, rotation=0)
    plt.plot(epochs, roc_aucs, lw=2, label='AUC')
    plt.plot(epochs, fpr, lw=2, label='Best FPR')
    plt.plot(epochs, acc, lw=2, label='Best Acc')
    plt.plot(epochs, recall, lw=2, label='Best Recall')
    plt.plot(epochs, precision, lw=2, label='Best Precision')
    plt.plot(epochs, fscore, lw=2, label='Best F-score')
    
    # plt.xlim([-0.01, 1.01])
    plt.xticks(epochs)
    # plt.ylim([-0.01, 1.01])
    plt.xlabel('Epochs')
    # plt.ylabel('AUC')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, len(epochs) + 1, 2)) 
    if dir is not None:
        plt.savefig(dir + f'/{file_name}.pdf', bbox_inches='tight')
        plt.savefig(dir + f'/{file_name}.eps', format='eps', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    max_auc = max(roc_aucs)
    opt_epoch = epochs[roc_aucs.index(max_auc)]
    opt_epoch -= 1
    print(f'Max AUC: {max_auc}, \nBest FPR: {fpr[opt_epoch]}, \nBest Acc: {acc[opt_epoch]}, \
        \nBest Recall: {recall[opt_epoch]}, \nBest Precision: {precision[opt_epoch]}, \nBest F-score: {fscore[opt_epoch]}, \
        \nOptimum number of epochs: {opt_epoch} ')
    
    if sw is not None:
        roc_aucs = roc_aucs
        fpr = fpr
        acc = acc
        recall = recall
        precision = precision
        fscore = fscore
        
        # plt.rcParams.update({'font.size': 18})
        plt.xticks(fontsize=10, rotation=0)
        plt.plot(epochs, roc_aucs, lw=2, label='AUC')
        plt.plot(epochs, fpr, lw=2, label='Best FPR')
        plt.plot(epochs, acc, lw=2, label='Best Acc')
        plt.plot(epochs, recall, lw=2, label='Best Recall')
        plt.plot(epochs, precision, lw=2, label='Best Precision')
        plt.plot(epochs, fscore, lw=2, label='Best F-score')

        # plt.xlim([-0.01, 1.01])
        plt.xticks(epochs)
        # plt.ylim([-0.01, 1.01])
        plt.xlabel('Epochs')
        # plt.ylabel('AUC')
        plt.title(title)
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.xticks(np.arange(0, len(epochs) + 1, 2)) 
        if dir is not None:
            plt.savefig(dir + f'/{file_name}_avg.pdf', bbox_inches='tight')
            plt.savefig(dir + f'/{file_name}_avg.eps', format='eps', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
    # plt.rcParams.update({'font.size': 20})


def plot_error_bar (x, y, yerr, log_path, file_name = '', xlim = None, ylim = None, logy = 0, logx = 0, save_eps = 0, ticks = None, ax = None, save_pdf = 1, plt_show = 1, axis_off = 0, plot_vertical_lines = 0):

    # Desired figure and axes size (in inches)
    fig_width_inches = 6.4
    fig_height_inches = 4.8
    axes_width_inches = 4.96
    axes_height_inches = 3.5


    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches)) # default size

    plt.errorbar(x, y, yerr=yerr, capsize=3, fmt="r--o", ecolor = "black")
    # plt.grid()

    if plot_vertical_lines:
        for _ in x:
            plt.axvline(_, color='black', linestyle = '--', linewidth = 1, alpha = 0.5)

    if logy:
        plt.yscale('log', nonposy='clip')
    if logx:
        plt.xscale('log')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if ticks is not None:
        plt.xticks(ticks)  # Generate ticks from 1 to 10, inclusive, with a step of 1
 

    if axis_off:
        plt.xticks([], [])
        plt.yticks([], [])
    
    # adjust axes size
    # Initialize the loop variables
    current_axes_width = 0
    current_axes_height = 0
    delta = 0.001
    left, right, bottom, top = 0, 1, 0, 1
    # Loop until the desired axes size is met
    while abs(current_axes_width - axes_width_inches) > 0.01 or abs(current_axes_height - axes_height_inches) > 0.01:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        
        ax = plt.gca()
        ax_position = ax.get_position()
        
        current_axes_width = fig_width_inches * (ax_position.x1 - ax_position.x0)
        current_axes_height = fig_height_inches * (ax_position.y1 - ax_position.y0)
        
        if current_axes_width > axes_width_inches:
            right -= delta
        elif current_axes_width < axes_width_inches:
            right += delta
            
        if current_axes_height > axes_height_inches:
            top -= delta
        elif current_axes_height < axes_height_inches:
            top += delta

    if save_pdf:
        plt.savefig(log_path + f'/{file_name}.pdf', bbox_inches='tight')
    if save_eps:
        plt.savefig(log_path + f'/{file_name}.eps', format='eps', bbox_inches='tight')
    if plt_show:
        plt.show()