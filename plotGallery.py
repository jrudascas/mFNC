import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from nilearn import plotting
import nibabel as nib
from scipy import stats

def plot1():
    from nilearn import datasets
    import nibabel as nib
    from scipy import stats
    import numpy as np
    from nilearn import plotting

    fMRI = nib.load(
        '/home/jrudascas/Desktop/Projects/Dataset/Original/Control/Zeevaert/components/4DficaAna_sub01_component_ica_s1_001.nii')
    fMRIData = fMRI.get_data()
    for index in range(fMRI.shape[-1]):
        print(index)
        if -fMRIData[:, :, :, index].min() > fMRIData[:, :, :, index].max():
            fMRIData[:, :, :, index] = - fMRIData[:, :, :, index]

        ic_threshold = stats.scoreatpercentile(np.abs(fMRIData[:, :, :, index]), 90)

        plotting.plot_glass_brain(nib.Nifti1Image(fMRIData[:, :, :, index], affine=fMRI.get_affine()),
                                  threshold=ic_threshold + 0.5, colorbar=True)
        plotting.plot_stat_map(nib.Nifti1Image(fMRIData[:, :, :, index], affine=fMRI.get_affine()),
                               threshold=ic_threshold, title="t-map, dim=-.5", dim=-.5)
        # plotting.plot_stat_map(fMRI, threshold=0.5, title="t-map, dim=-.5", dim=-.5, annotate=True)
        plotting.show()

    a = 3

    localizer_dataset = datasets.fetch_localizer_button_task(get_anats=True)
    localizer_anat_filename = localizer_dataset.anats[0]
    localizer_tmap_filename = localizer_dataset.tmaps[0]

    # localizer_anat_filename = nib.load('/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_2mm_brain.nii.gz')
    # T1 = localizer_anat_filename.get_data()

    fMRI = nib.load(
        '/home/jrudascas/Desktop/Projects/Dataset/Original/Control/Zeevaert/components/4DficaAna_sub01_component_ica_s1_001.nii')
    fMRIData = fMRI.get_data()

    # for index in range(fMRI.shape[-1]):
    #    print(index)
    if -fMRIData[:, :, :, 4].min() > fMRIData[:, :, :, 4].max():
        fMRIData[:, :, :, 4] = - fMRIData[:, :, :, 4]

    ic_threshold = stats.scoreatpercentile(np.abs(fMRIData[:, :, :, 4]), 90)

    plotting.plot_glass_brain(
        nib.Nifti1Image(fMRIData[:, :, :, 4], affine=fMRI.get_affine()),
        threshold=0.6, colorbar=True)

    print(ic_threshold)

    if -fMRIData[:, :, :, 24].min() > fMRIData[:, :, :, 24].max():
        fMRIData[:, :, :, 24] = - fMRIData[:, :, :, 24]

    ic_threshold = stats.scoreatpercentile(np.abs(fMRIData[:, :, :, 24]), 90)
    print(ic_threshold)
    plotting.plot_glass_brain(
        nib.Nifti1Image(fMRIData[:, :, :, 24], affine=fMRI.get_affine()),
        threshold=0.5, colorbar=True)

    # plotting.plot_stat_map(fMRI, threshold=0.5, title="t-map, dim=-.5", dim=-.5, annotate=True)
    plotting.show()


def barchart(group1, group2, title = "Group 1 vs Group 2", labelGroup1="Group 1", labelGroup2 = "Group 2", xLabel = "x", yLabel = "y", labelFeautures = None, outlier = None, save = None):
    n_feautures = group1.shape[1]

    if labelFeautures is None:
        labelFeautures = np.array(range(n_feautures)) + 1

    means_group1 = np.zeros(n_feautures)
    std_group1 = np.zeros(n_feautures)
    means_group2 = np.zeros(n_feautures)
    std_group2 = np.zeros(n_feautures)

    if outlier is None:
        means_group1 = np.mean(group1, axis=0)
        std_group1 = np.std(group1, axis=0)

        means_group2 = np.mean(group2, axis=0)
        std_group2 = np.std(group2, axis=0)
    else:
        for index in range(n_feautures):
            means_group1[index] = np.mean(group1[(group1[:,index] != outlier), index])
            std_group1[index] = np.var(group1[(group1[:,index] != outlier), index])
            means_group2[index] = np.mean(group2[(group2[:,index] != outlier), index])
            std_group2[index] = np.var(group2[(group2[:,index] != outlier), index])

    fig, ax = plt.subplots()

    index = np.arange(n_feautures)
    bar_width = 0.25

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, height=means_group1, width=bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=std_group1,
                     error_kw=error_config,
                     label=labelGroup1)

    rects2 = plt.bar(index + bar_width, height=means_group2, width=bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=std_group2,
                     error_kw=error_config,
                     label=labelGroup2)
    import matplotlib.font_manager as fm

    plt.xlabel(xLabel, fontsize=10)
    plt.ylabel(yLabel, fontsize=10)

    plt.title(title, fontsize=12)
    plt.xticks(index, labelFeautures, rotation='vertical', fontsize=7)
    plt.legend(loc = 'upper right', fontsize=10)

    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300)
    plt.show()

def barchart2(group1, group2, group3, group4, title = "Group 1 vs Group 2", labelGroup1="Group 1", labelGroup2 = "Group 2", labelGroup3="Group 3", labelGroup4 = "Group 4", xLabel = "x", yLabel = "y", labelFeautures = None, outlier = None):
    n_feautures = group1.shape[1]

    if labelFeautures is None:
        labelFeautures = np.array(range(n_feautures)) + 1

    means_group1 = np.zeros(n_feautures)
    std_group1 = np.zeros(n_feautures)
    means_group2 = np.zeros(n_feautures)
    std_group2 = np.zeros(n_feautures)
    means_group3 = np.zeros(n_feautures)
    std_group3 = np.zeros(n_feautures)
    means_group4 = np.zeros(n_feautures)
    std_group4 = np.zeros(n_feautures)

    if outlier is None:
        means_group1 = np.mean(group1, axis=0)
        std_group1 = np.std(group1, axis=0)

        means_group2 = np.mean(group2, axis=0)
        std_group2 = np.std(group2, axis=0)

        means_group3 = np.mean(group3, axis=0)
        std_group3 = np.std(group3, axis=0)

        means_group4 = np.mean(group4, axis=0)
        std_group4 = np.std(group4, axis=0)
    else:
        for index in range(n_feautures):
            means_group1[index] = np.mean(group1[(group1[:,index] != outlier), index])
            std_group1[index] = np.var(group1[(group1[:,index] != outlier), index])
            means_group2[index] = np.mean(group2[(group2[:,index] != outlier), index])
            std_group2[index] = np.var(group2[(group2[:,index] != outlier), index])

            means_group3[index] = np.mean(group3[(group3[:, index] != outlier), index])
            std_group3[index] = np.var(group3[(group3[:, index] != outlier), index])
            means_group4[index] = np.mean(group4[(group4[:, index] != outlier), index])
            std_group4[index] = np.var(group4[(group4[:, index] != outlier), index])

    fig, ax = plt.subplots()

    index = np.arange(n_feautures)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, height=means_group1, width=bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=std_group1,
                     error_kw=error_config,
                     label=labelGroup1)

    rects2 = plt.bar(index + bar_width, height=means_group2, width=bar_width,
                     alpha=opacity,
                     color='aqua',
                     yerr=std_group2,
                     error_kw=error_config,
                     label=labelGroup2)

    rects3 = plt.bar(index + 2*bar_width, height=means_group3, width=bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=std_group3,
                     error_kw=error_config,
                     label=labelGroup3)

    rects4 = plt.bar(index + 3*bar_width, height=means_group4, width=bar_width,
                     alpha=opacity,
                     color='orangered',
                     yerr=std_group4,
                     error_kw=error_config,
                     label=labelGroup4)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.xticks(index + 2*bar_width, labelFeautures, rotation='vertical')
    #plt.legend()

    #plt.tight_layout()
    plt.show()

def barchart3(group1, group2, group3, group4, title = "Group 1 vs Group 2", labelGroup1="Group 1", labelGroup2 = "Group 2", labelGroup3="Group 3", labelGroup4 = "Group 4", labelGroup5="Group 5", labelGroup6 = "Group 6", labelGroup7="Group 7", labelGroup8 = "Group 8", xLabel = "x", yLabel = "y", labelFeautures = None, outlier = None):
    n_feautures = group1.shape[-1]

    if labelFeautures is None:
        labelFeautures = np.array(range(n_feautures)) + 1

    means_group1 = np.zeros(n_feautures)
    std_group1 = np.zeros(n_feautures)
    means_group2 = np.zeros(n_feautures)
    std_group2 = np.zeros(n_feautures)
    means_group3 = np.zeros(n_feautures)
    std_group3 = np.zeros(n_feautures)
    means_group4 = np.zeros(n_feautures)
    std_group4 = np.zeros(n_feautures)

    if outlier is None:
        means_group1 = np.mean(group1, axis=0)
        std_group1 = np.std(group1, axis=0)

        means_group2 = np.mean(group2, axis=0)
        std_group2 = np.std(group2, axis=0)

        means_group3 = np.mean(group3, axis=0)
        std_group3 = np.std(group3, axis=0)

        means_group4 = np.mean(group4, axis=0)
        std_group4 = np.std(group4, axis=0)
    else:
        for index in range(n_feautures):
            means_group1[index] = np.mean(group1[(group1[:,index] != outlier), index])
            std_group1[index] = np.std(group1[(group1[:,index] != outlier), index])
            means_group2[index] = np.mean(group2[(group2[:,index] != outlier), index])
            std_group2[index] = np.std(group2[(group2[:,index] != outlier), index])

            means_group3[index] = np.mean(group3[(group3[:, index] != outlier), index])
            std_group3[index] = np.std(group3[(group3[:, index] != outlier), index])
            means_group4[index] = np.mean(group4[(group4[:, index] != outlier), index])
            std_group4[index] = np.std(group4[(group4[:, index] != outlier), index])


    fig, ax = plt.subplots()

    index = np.arange(4)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(1, height=np.mean(means_group1), width=bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=np.std(means_group1),
                     error_kw=error_config,
                     label=labelGroup1)

    rects2 = plt.bar(1 + bar_width, height=np.mean(means_group2), width=bar_width,
                     alpha=opacity,
                     color='aqua',
                     yerr=np.std(means_group2),
                     error_kw=error_config,
                     label=labelGroup2)

    rects3 = plt.bar(2, height=np.mean(means_group3), width=bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=np.std(means_group3),
                     error_kw=error_config,
                     label=labelGroup3)

    rects4 = plt.bar(2 + bar_width, height=np.mean(means_group4), width=bar_width,
                     alpha=opacity,
                     color='orangered',
                     yerr=np.std(means_group4),
                     error_kw=error_config,
                     label=labelGroup4)

    rects5 = plt.bar(3, height=np.mean(std_group1), width=bar_width,
                     alpha=opacity,
                     color='purple',
                     yerr=np.std(std_group1),
                     error_kw=error_config,
                     label=labelGroup5)

    rects6 = plt.bar(3 + bar_width, height=np.mean(std_group2), width=bar_width,
                     alpha=opacity,
                     color='hotpink',
                     yerr=np.std(std_group2),
                     error_kw=error_config,
                     label=labelGroup6)

    rects7 = plt.bar(4, height=np.mean(std_group3), width=bar_width,
                     alpha=opacity,
                     color='gold',
                     yerr=np.std(std_group3),
                     error_kw=error_config,
                     label=labelGroup7)

    rects8 = plt.bar(4 + bar_width, height=np.mean(std_group4), width=bar_width,
                     alpha=opacity,
                     color='goldenrod',
                     yerr=np.std(std_group4),
                     error_kw=error_config,
                     label=labelGroup8)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.xticks(index + 1.2, labelFeautures)
    plt.legend()

    plt.tight_layout()
    plt.show()


def barchart4(group1, group2, group3, group4, title = "Group 1 vs Group 2", labelGroup1="Group 1", labelGroup2 = "Group 2", labelGroup3="Group 3", labelGroup4 = "Group 4", xLabel = "x", yLabel = "y", labelFeautures = None, outlier = None, save = None):
    n_feautures = group1.shape[0]

    if labelFeautures is None:
        labelFeautures = np.array(range(n_feautures)) + 1

    means_group1 = np.zeros(n_feautures)
    means_group2 = np.zeros(n_feautures)
    means_group3 = np.zeros(n_feautures)
    means_group4 = np.zeros(n_feautures)
    x = np.zeros(n_feautures)

    if outlier is None:
        means_group1 = np.mean(group1, axis=0)
        means_group2 = np.mean(group2, axis=0)
        means_group3 = np.mean(group3, axis=0)
        means_group4 = np.mean(group4, axis=0)

    else:
        for index in range(n_feautures):
            if np.alltrue(group1[index,:] == outlier):
                means_group1[index] = np.nan
            else:
                means_group1[index] = np.mean(group1[index, (group1[index,:] != outlier)])

            if np.alltrue(group2[index,:] == outlier):
                means_group2[index] = np.nan
            else:
                means_group2[index] = np.mean(group2[index, (group2[index,:] != outlier)])

            if np.alltrue(group3[index,:] == outlier):
                means_group3[index] = np.nan
            else:
                means_group3[index] = np.mean(group3[index, (group3[index,:] != outlier)])

            if np.alltrue(group4[index,:] == outlier):
                means_group4[index] = np.nan
            else:
                means_group4[index] = np.mean(group4[index, (group4[index,:] != outlier)])

            #x[index] = np.sum(group1[index, (group1[index, :] != outlier)]) / len((group1[index, (group1[index, :] != outlier)]))

    means_group1 = means_group1[~np.isnan(means_group1)]
    means_group2 = means_group2[~np.isnan(means_group2)]
    means_group3 = means_group3[~np.isnan(means_group3)]
    means_group4 = means_group4[~np.isnan(means_group4)]

    fig, ax = plt.subplots()

    index = np.arange(2)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(1, height=np.mean(means_group1), width=bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=np.var(means_group1),
                     error_kw=error_config,
                     label=labelGroup1)

    rects2 = plt.bar(1 + bar_width, height=np.mean(means_group2), width=bar_width,
                     alpha=opacity,
                     color='aqua',
                     yerr=np.var(means_group2),
                     error_kw=error_config,
                     label=labelGroup2)

    rects3 = plt.bar(2, height=np.mean(means_group3), width=bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=np.var(means_group3),
                     error_kw=error_config,
                     label=labelGroup3)

    rects4 = plt.bar(2 + bar_width, height=np.mean(means_group4), width=bar_width,
                     alpha=opacity,
                     color='orangered',
                     yerr=np.var(means_group4),
                     error_kw=error_config,
                     label=labelGroup4)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.xticks(index + 1.2, labelFeautures)
    plt.legend()

    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300)
    plt.show()



def fivethirtyeightPlot(group1, group2, group3 = None, lag = 0, labelFeautures=None, save = None):

    new = np.zeros((lag * 2 + 1, group1.shape[1]))
    new2 = np.zeros((lag * 2 + 1, group2.shape[1]))

    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            new[i, j] = len(np.where(group1[:, j] == i - lag)[0])/group1.shape[0]

    for i in range(new2.shape[0]):
        for j in range(new2.shape[1]):
            new2[i, j] = len(np.where(group2[:, j] == i - lag)[0])/group2.shape[0]

    dpi = 300

    # plot violin plot
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))


    if group3 is not None:
        new3 = np.zeros((lag * 2 + 1, group3.shape[1]))

        for i in range(new3.shape[0]):
            for j in range(new3.shape[1]):
                new3[i, j] = len(np.where(group3[:, j] == i - lag)[0]) / group3.shape[0]

        axes.violinplot(np.transpose(new3), positions=[1, 5, 9, 13, 17, 21, 25],
                        showmeans=True,
                        showextrema=True)

        axes.violinplot(np.transpose(new), positions=[2, 6, 10, 14, 18, 22, 26],
                        showmeans=True,
                        showmedians=False)

        axes.violinplot(np.transpose(new2), positions=[3, 7, 11, 15, 19, 23, 27],
                        showmeans=True,
                        showextrema=True)
    else:
        axes.violinplot(np.transpose(new), positions=[1, 4, 7, 10, 13, 16, 19],
                        showmeans=True,
                        showmedians=False)

        axes.violinplot(np.transpose(new2), positions=[2, 5, 8, 11, 14, 17, 20],
                        showmeans=True,
                        showextrema=True)

    axes.set_title('violin plot')
    plt.show()


    fig, ax = plt.subplots(figsize = (2000 / dpi, 3000 / dpi))
    plt.subplots_adjust(bottom=0.33)

    ax.grid(b=False)
    #ax.axis('off')

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    scale = 2.5
    for index in range(new.shape[0]):
        ax.plot(scale*new[index, :] + (index*1.5) + index, color='red', linewidth=2.0)
        ax.plot(scale*new2[index, :] + (index*1.5) + index + 1.2, color='blue', linewidth=2.0)

        #plt.text(-2, (index*1.5) + index, str((index*1.5) + index))
        #plt.text(-2, (index*1.5) + index + 1.2, str((index*1.5) + index + 1.2))
        #plt.text(-2, (index * 1.5) + index - 0.1, '0%', fontsize=6)
        #plt.text(-2, (index * 1.5) + index + 1 - 0.1, '100%', fontsize=6)

        #plt.text(-2, (index * 1.5) + index + 1.2,'0%', fontsize=6)
        #plt.text(-2, (index * 1.5) + index + 1.0 + 1, '100%', fontsize=6)

        ax.plot(np.linspace(np.mean(scale*new[index, :]) + (index*1.5) + index, np.mean(scale*new[index, :]) + (index*1.5) + index, new.shape[1]), alpha=0.8, color='black', linewidth=1.0, linestyle= '--')
        ax.plot(np.linspace(np.mean(scale*new2[index, :]) + (index*1.5) + index + 1.2, np.mean(scale*new2[index, :]) + (index*1.5) + index + 1.2, new2.shape[1]), alpha=0.8, color='black', linewidth=1.0, linestyle= '--')

        #ax.plot(np.linspace(np.mean(new2[index, :]) + (index * 1.5) + index + 2.2, np.mean(new2[index, :]) + (index * 1.5) + index + 2.2, new2.shape[1]), alpha=0.4, color='b', linewidth=1.0, linestyle='--')

        plt.text(46, np.mean(scale*new[index, :]) + (index*1.5) + index - 0.1, str(round(100*np.mean(new[index, :]))) + '%', fontsize=10)
        plt.text(46, np.mean(scale*new2[index, :]) + (index*1.5) + index + 1.2 - 0.1, str(round(100*np.mean(new2[index, :]))) + '%', fontsize=10)

        import utils


    if labelFeautures is None:
        labelFeautures = np.array(range(new2.shape[1])) + 1

    plt.xticks(range(new2.shape[1]), labelFeautures, rotation='vertical', fontsize=7)

    print("MCS vs UWS")
    print(utils.toFindStatisticDifference(np.transpose(new), np.transpose(new2), measure='manwhitneyu'))

    print("HC vs MCS")
    print(utils.toFindStatisticDifference(np.transpose(new3), np.transpose(new), measure='manwhitneyu'))

    print("HC vs UWS")
    print(utils.toFindStatisticDifference(np.transpose(new3), np.transpose(new2), measure='manwhitneyu'))


    #if save is not None:
        #fig.savefig(save, dpi=dpi)



    plt.show()


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta = theta*-1
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def plotRSN():
    mainPath = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
    lastPath = 'Allain/components/4DicaAna_sub01_component_ica_s1_001.nii.gz'

    fMRI = nib.load(mainPath + lastPath)
    fMRIData = fMRI.get_data()

    print()
    for index in range(fMRI.shape[-1]):
        print(index)
        if -fMRIData[:, :, :, index].min() > fMRIData[:, :, :, index].max():
            fMRIData[:, :, :, index] = - fMRIData[:, :, :, index]

        ic_threshold = stats.scoreatpercentile(np.abs(fMRIData[:, :, :, index]), 90)

        #plotting.plot_glass_brain(nib.Nifti1Image(fMRIData[:, :, :, index], affine=fMRI.get_affine()),
    #                              threshold=ic_threshold + 0.5, colorbar=True)
        plotting.plot_stat_map(nib.Nifti1Image(fMRIData[:, :, :, index], affine=fMRI.get_affine()),
                               threshold=ic_threshold, dim=-.5)
                               #threshold=ic_threshold, dim=-.5, output_file=mainPath + 'imgComponent_' + str(index+1) + ".png")
                               #threshold=ic_threshold, dim=-.5, output_file=str(index) + ".png")
        # plotting.plot_stat_map(fMRI, threshold=0.5, title="t-map, dim=-.5", dim=-.5, annotate=True)
        plotting.show()



def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See  doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    data = [
        [],
        ('Mean connectivity level - Lagged', [
            [0.069420770386, 0.00573539525575, 0.0162095387885, 0.111718033662, 0.107313443794, 0.149935833742, 0.217244247237, 0.0579392764386, 0.249118911229, 0.103086482682],
            [0.161831272388,0.0806519174197,0.0324767297485,0.0413598139312,0.0149021466021,0.0586738147275,0.0285511288457,0.0571931958619,0.025518564839,0.00139978224678],
            [0.150103057145,0.368477597194,0.202262656941,0.331083935473,0.269287087156,0.016714522278,0.150772784627,0.167648572874,0.285004669823,0.113104747768]]),

        ('Mean connectivity level - Non-Lagged', [
            [0.0971660835114,0.0894232049314,0.0301995900642,0.0986390414741,0.109788595315,0.145767062967,0.233062017493,0.0394373752283,0.236630719268,0.12059094796],
            [0.144946127911,0.0374530413889,0.036083470618,0.0739375560835,0.021326841347,0.0843846117338,0.0537457413819,0.0853897297377,0.00347483925532,0.0125749676474],
            [0.0851575650883,0.110585142991,0.121977419371,0.246648176979,0.27257664366,0.000652032558407,0.0580258460568,0.16958400677,0.22297361671,0.120792212983]]),

        ('Hyperconnectivity Counting - Lagged', [
            [2.25,1.71428571429,2.32,1.33333333333,2.34782608696,2.85714285714,3.0,2.8,2.17391304348,0.866666666667],
            [2.5,2.4,2.33333333333,2.75,2.83333333333,2.16666666667,2.83333333333,3.33333333333,2.54545454545,1.375],
            [3.42857142857,4.2,5.16666666667,4.625,4.625,3.83333333333,4.11111111111,3.625,4.14285714286,1.14285714286]]),
        ('Hyperconnectivity Counting - Non-Lagged', [
            [1.41666666667,0.571428571429,1.2,1.0,1.34782608696,1.28571428571,2.33333333333,1.8,1.34782608696,0.733333333333],
            [1.375,1.6,1.11111111111,1.33333333333,1.66666666667,1.33333333333,1.83333333333,2.0,1.36363636364,0.5],
            [2.14285714286,3.2,3.66666666667,3.125,3.625,2.0,2.77777777778,1.875,2.71428571429,0.857142857143]])
    ]
    return data

def example_data2():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See  doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    data = [
        ['Auditory', 'Cerebellum', 'DMN', 'ECL', 'ECR', 'Salience', 'SensoriMotor', 'Vis_Lateral', 'Vis_Medial', 'Vis_Occipital'],
        ('', [
            [70,	80,	70,	30,	40,	40,	60,	50,	50,	60],
            [100,	30,	20,	10,	0,	40,	40,	10,	10,	60],
            [30,	20,	10,	20,	50,	70,	30,	10,	30,	30],
            [70,	30,	30,	10,	30,	60,	40,	20,	20,	50]])
    ]
    return data

if __name__ == '__main__':
    N = 10
    theta = radar_factory(N, frame='polygon')

    data = example_data2()
    spoke_labels = data.pop(0)
    dpi = 300
    fig, axes = plt.subplots(figsize=(4500 / dpi, 4500 / dpi), nrows=1, ncols=2, subplot_kw=dict(projection='radar'))
    #fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes

    for ax, (title, case_data) in zip(axes.flatten(), data):
        ax.set_rgrids([20,40,60,80,100])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    #ax = axes[0, 0]
    labels = ('Evaluator # 1','Evaluator # 2', 'Evaluator # 3', 'Consensus')
    legend = ax.legend(labels, loc=(1.1, .95), labelspacing=0.1, fontsize='small')

    #fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios', horizontalalignment='center', color='black', weight='bold', size='large')

    fig.savefig('radar3.png', dpi=300)

    plt.show()