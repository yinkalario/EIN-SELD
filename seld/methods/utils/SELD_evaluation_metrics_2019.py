#
# Implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and
# The DOA metrics are explained in the SELDnet paper
#
# This script has MIT license
#

import numpy as np
from scipy.optimize import linear_sum_assignment
from IPython import embed
eps = np.finfo(np.float).eps


##########################################################################################
# SELD scoring functions - class implementation
#
# NOTE: Supports only one-hot labels for both SED and DOA. Doesnt work for baseline method
# directly, since it estimated DOA in regression approach. Check below the class for
# one shot (function) implementations of all metrics. The function implementation has
# support for both one-hot labels and regression values of DOA estimation.
##########################################################################################

class SELDMetrics(object):
    def __init__(self, nb_frames_1s=None, data_gen=None):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0
        self._block_size = nb_frames_1s

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._data_gen = data_gen

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0

    def f1_overall_framewise(self, O, T):
        TP = ((2 * T - O) == 1).sum()
        Nref, Nsys = T.sum(), O.sum()
        self._TP += TP
        self._Nref += Nref
        self._Nsys += Nsys

    def er_overall_framewise(self, O, T):
        FP = np.logical_and(T == 0, O == 1).sum(1)
        FN = np.logical_and(T == 1, O == 0).sum(1)
        S = np.minimum(FP, FN).sum()
        D = np.maximum(0, FN - FP).sum()
        I = np.maximum(0, FP - FN).sum()
        self._S += S
        self._D += D
        self._I += I

    def f1_overall_1sec(self, O, T):        
        new_size = int(np.ceil(float(O.shape[0]) / self._block_size))
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.f1_overall_framewise(O_block, T_block)

    def er_overall_1sec(self, O, T):        
        new_size = int(np.ceil(float(O.shape[0]) / self._block_size))
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.er_overall_framewise(O_block, T_block)

    def update_sed_scores(self, pred, gt):
        """
        Computes SED metrics for one second segments

        :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param nb_frames_1s: integer, number of frames in one second
        :return:
        """
        self.f1_overall_1sec(pred, gt)
        self.er_overall_1sec(pred, gt)

    def compute_sed_scores(self):
        ER = (self._S + self._D + self._I) / (self._Nref + 0.0)

        prec = float(self._TP) / float(self._Nsys + eps)
        recall = float(self._TP) / float(self._Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        return ER, F

    def update_doa_scores(self, pred_doa_thresholded, gt_doa):
        '''
        Compute DOA metrics when DOA is estimated using classification approach

        :param pred_doa_thresholded: predicted results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                                    with value 1 when sound event active, else 0
        :param gt_doa: reference results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                        with value 1 when sound event active, else 0
        :param data_gen_test: feature or data generator class

        :return: DOA metrics

        '''
        self._doa_loss_pred_cnt += np.sum(pred_doa_thresholded)
        self._nb_frames += pred_doa_thresholded.shape[0]

        for frame in range(pred_doa_thresholded.shape[0]):
            nb_gt_peaks = int(np.sum(gt_doa[frame, :]))
            nb_pred_peaks = int(np.sum(pred_doa_thresholded[frame, :]))

            # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
            if nb_gt_peaks == nb_pred_peaks:
                self._nb_good_pks += 1
            elif nb_gt_peaks > nb_pred_peaks:
                self._less_est_frame_cnt += 1
                self._less_est_cnt += (nb_gt_peaks - nb_pred_peaks)
            elif nb_pred_peaks > nb_gt_peaks:
                self._more_est_frame_cnt += 1
                self._more_est_cnt += (nb_pred_peaks - nb_gt_peaks)

            # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
            # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
            if nb_gt_peaks and nb_pred_peaks:
                pred_ind = np.where(pred_doa_thresholded[frame] == 1)[1]
                pred_list_rad = np.array(self._data_gen .get_matrix_index(pred_ind)) * np.pi / 180

                gt_ind = np.where(gt_doa[frame] == 1)[1]
                gt_list_rad = np.array(self._data_gen .get_matrix_index(gt_ind)) * np.pi / 180

                frame_dist = distance_between_gt_pred(gt_list_rad.T, pred_list_rad.T)
                self._doa_loss_pred += frame_dist

    def compute_doa_scores(self):
        doa_error = self._doa_loss_pred / self._doa_loss_pred_cnt
        frame_recall = self._nb_good_pks / float(self._nb_frames)
        return doa_error, frame_recall

    def reset(self):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0


###############################################################
# SED scoring functions
###############################################################


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)

    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER


def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(float(O.shape[0]) / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(float(O.shape[0]) / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_sed_scores(pred, gt, nb_frames_1s):
    """
    Computes SED metrics for one second segments

    :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param nb_frames_1s: integer, number of frames in one second
    :return:
    """
    f1o = f1_overall_1sec(pred, gt, nb_frames_1s)
    ero = er_overall_1sec(pred, gt, nb_frames_1s)
    scores = [ero, f1o]
    return scores


###############################################################
# DOA scoring functions
###############################################################


def compute_doa_scores_regr_xyz(pred_doa, gt_doa, pred_sed, gt_sed):
    """
        Compute DOA metrics when DOA is estimated using regression approach

    :param pred_doa: predicted doa_labels is of dimension [nb_frames, 3*nb_classes],
                        nb_classes each for x, y, and z axes,
                        if active, the DOA values will be in real numbers [-1 1] range, else, it will contain default doa values of (0, 0, 0)
    :param gt_doa: reference doa_labels is of dimension [nb_frames, 3*nb_classes],
    :param pred_sed: predicted sed label of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
    :param gt_sed: reference sed label of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
    :return:
    """

    nb_src_gt_list = np.zeros(gt_doa.shape[0]).astype(int)
    nb_src_pred_list = np.zeros(gt_doa.shape[0]).astype(int)
    good_frame_cnt = 0
    doa_loss_pred = 0.0
    nb_sed = gt_sed.shape[-1]

    less_est_cnt, less_est_frame_cnt = 0, 0
    more_est_cnt, more_est_frame_cnt = 0, 0

    for frame_cnt, sed_frame in enumerate(gt_sed):
        nb_src_gt_list[frame_cnt] = int(np.sum(sed_frame))
        nb_src_pred_list[frame_cnt] = int(np.sum(pred_sed[frame_cnt]))

        # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
        if nb_src_gt_list[frame_cnt] == nb_src_pred_list[frame_cnt]:
            good_frame_cnt = good_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] > nb_src_pred_list[frame_cnt]:
            less_est_cnt = less_est_cnt + nb_src_gt_list[frame_cnt] - nb_src_pred_list[frame_cnt]
            less_est_frame_cnt = less_est_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] < nb_src_pred_list[frame_cnt]:
            more_est_cnt = more_est_cnt + nb_src_pred_list[frame_cnt] - nb_src_gt_list[frame_cnt]
            more_est_frame_cnt = more_est_frame_cnt + 1

        # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
        # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
        if nb_src_gt_list[frame_cnt] and nb_src_pred_list[frame_cnt]:
            # DOA Loss with respect to predicted confidence
            sed_frame_gt = gt_sed[frame_cnt]
            doa_frame_gt_x = gt_doa[frame_cnt][:nb_sed][sed_frame_gt == 1]
            doa_frame_gt_y = gt_doa[frame_cnt][nb_sed:2*nb_sed][sed_frame_gt == 1]
            doa_frame_gt_z = gt_doa[frame_cnt][2*nb_sed:][sed_frame_gt == 1]

            sed_frame_pred = pred_sed[frame_cnt]
            doa_frame_pred_x = pred_doa[frame_cnt][:nb_sed][sed_frame_pred == 1]
            doa_frame_pred_y = pred_doa[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]
            doa_frame_pred_z = pred_doa[frame_cnt][2*nb_sed:][sed_frame_pred == 1]

            doa_loss_pred += distance_between_gt_pred_xyz(np.vstack((doa_frame_gt_x, doa_frame_gt_y, doa_frame_gt_z)).T,
                                                      np.vstack((doa_frame_pred_x, doa_frame_pred_y, doa_frame_pred_z)).T)

    doa_loss_pred_cnt = np.sum(nb_src_pred_list)
    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    frame_recall = good_frame_cnt / float(gt_sed.shape[0])
    er_metric = [doa_loss_pred, frame_recall, doa_loss_pred_cnt, good_frame_cnt, more_est_cnt, less_est_cnt]
    return er_metric


def compute_doa_scores_regr(pred_doa_rad, gt_doa_rad, pred_sed, gt_sed):
    """
        Compute DOA metrics when DOA is estimated using regression approach

    :param pred_doa_rad: predicted doa_labels is of dimension [nb_frames, 2*nb_classes],
                        nb_classes each for azimuth and elevation angles,
                        if active, the DOA values will be in RADIANS, else, it will contain default doa values
    :param gt_doa_rad: reference doa_labels is of dimension [nb_frames, 2*nb_classes],
                    nb_classes each for azimuth and elevation angles,
                    if active, the DOA values will be in RADIANS, else, it will contain default doa values
    :param pred_sed: predicted sed label of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
    :param gt_sed: reference sed label of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
    :return:
    """

    nb_src_gt_list = np.zeros(gt_doa_rad.shape[0]).astype(int)
    nb_src_pred_list = np.zeros(gt_doa_rad.shape[0]).astype(int)
    good_frame_cnt = 0
    doa_loss_pred = 0.0
    nb_sed = gt_sed.shape[-1]

    less_est_cnt, less_est_frame_cnt = 0, 0
    more_est_cnt, more_est_frame_cnt = 0, 0

    for frame_cnt, sed_frame in enumerate(gt_sed):
        nb_src_gt_list[frame_cnt] = int(np.sum(sed_frame))
        nb_src_pred_list[frame_cnt] = int(np.sum(pred_sed[frame_cnt]))

        # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
        if nb_src_gt_list[frame_cnt] == nb_src_pred_list[frame_cnt]:
            good_frame_cnt = good_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] > nb_src_pred_list[frame_cnt]:
            less_est_cnt = less_est_cnt + nb_src_gt_list[frame_cnt] - nb_src_pred_list[frame_cnt]
            less_est_frame_cnt = less_est_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] < nb_src_pred_list[frame_cnt]:
            more_est_cnt = more_est_cnt + nb_src_pred_list[frame_cnt] - nb_src_gt_list[frame_cnt]
            more_est_frame_cnt = more_est_frame_cnt + 1

        # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
        # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
        if nb_src_gt_list[frame_cnt] and nb_src_pred_list[frame_cnt]:
            # DOA Loss with respect to predicted confidence
            sed_frame_gt = gt_sed[frame_cnt]
            doa_frame_gt_azi = gt_doa_rad[frame_cnt][:nb_sed][sed_frame_gt == 1]
            doa_frame_gt_ele = gt_doa_rad[frame_cnt][nb_sed:][sed_frame_gt == 1]

            sed_frame_pred = pred_sed[frame_cnt]
            doa_frame_pred_azi = pred_doa_rad[frame_cnt][:nb_sed][sed_frame_pred == 1]
            doa_frame_pred_ele = pred_doa_rad[frame_cnt][nb_sed:][sed_frame_pred == 1]

            doa_loss_pred += distance_between_gt_pred(np.vstack((doa_frame_gt_azi, doa_frame_gt_ele)).T,
                                                      np.vstack((doa_frame_pred_azi, doa_frame_pred_ele)).T)

    doa_loss_pred_cnt = np.sum(nb_src_pred_list)
    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    frame_recall = good_frame_cnt / float(gt_sed.shape[0])
    er_metric = [doa_loss_pred, frame_recall, doa_loss_pred_cnt, good_frame_cnt, more_est_cnt, less_est_cnt]
    return er_metric


def compute_doa_scores_clas(pred_doa_thresholded, gt_doa, data_gen_test):
    '''
    Compute DOA metrics when DOA is estimated using classification approach

    :param pred_doa_thresholded: predicted results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                                with value 1 when sound event active, else 0
    :param gt_doa: reference results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                    with value 1 when sound event active, else 0
    :param data_gen_test: feature or data generator class

    :return: DOA metrics

    '''
    doa_loss_pred_cnt = np.sum(pred_doa_thresholded)

    doa_loss_pred = 0
    nb_good_pks = 0

    less_est_cnt, less_est_frame_cnt = 0, 0
    more_est_cnt, more_est_frame_cnt = 0, 0

    for frame in range(pred_doa_thresholded.shape[0]):
        nb_gt_peaks = int(np.sum(gt_doa[frame, :]))
        nb_pred_peaks = int(np.sum(pred_doa_thresholded[frame, :]))

        # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
        if nb_gt_peaks == nb_pred_peaks:
            nb_good_pks += 1
        elif nb_gt_peaks > nb_pred_peaks:
            less_est_frame_cnt += 1
            less_est_cnt += (nb_gt_peaks - nb_pred_peaks)
        elif nb_pred_peaks > nb_gt_peaks:
            more_est_frame_cnt += 1
            more_est_cnt += (nb_pred_peaks - nb_gt_peaks)

        # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
        # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
        if nb_gt_peaks and nb_pred_peaks:
            pred_ind = np.where(pred_doa_thresholded[frame] == 1)[1]
            pred_list_rad = np.array(data_gen_test.get_matrix_index(pred_ind)) * np.pi / 180

            gt_ind = np.where(gt_doa[frame] == 1)[1]
            gt_list_rad = np.array(data_gen_test.get_matrix_index(gt_ind)) * np.pi / 180

            frame_dist = distance_between_gt_pred(gt_list_rad.T, pred_list_rad.T)
            doa_loss_pred += frame_dist

    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    frame_recall = nb_good_pks / float(pred_doa_thresholded.shape[0])
    er_metric = [doa_loss_pred, frame_recall, doa_loss_pred_cnt, nb_good_pks, more_est_cnt, less_est_cnt]
    return er_metric


def distance_between_gt_pred(gt_list_rad, pred_list_rad):
    """
    Shortest distance between two sets of spherical coordinates. Given a set of groundtruth spherical coordinates,
     and its respective predicted coordinates, we calculate the spherical distance between each of the spherical
     coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
     coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
     groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
     least cost in this distance matrix.

    :param gt_list_rad: list of ground-truth spherical coordinates
    :param pred_list_rad: list of predicted spherical coordinates
    :return: cost -  distance
    :return: less - number of DOA's missed
    :return: extra - number of DOA's over-estimated
    """

    gt_len, pred_len = gt_list_rad.shape[0], pred_list_rad.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    # Slow implementation
    # cost_mat = np.zeros((gt_len, pred_len))
    # for gt_cnt, gt in enumerate(gt_list_rad):
    #     for pred_cnt, pred in enumerate(pred_list_rad):
    #         cost_mat[gt_cnt, pred_cnt] = distance_between_spherical_coordinates_rad(gt, pred)

    # Fast implementation
    if gt_len and pred_len:
        az1, ele1, az2, ele2 = gt_list_rad[ind_pairs[:, 0], 0], gt_list_rad[ind_pairs[:, 0], 1], \
                               pred_list_rad[ind_pairs[:, 1], 0], pred_list_rad[ind_pairs[:, 1], 1]
        cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind].sum()
    return cost


def distance_between_gt_pred_xyz(gt_list, pred_list):
    """
    Shortest distance between two sets of Cartesian coordinates. Given a set of groundtruth coordinates,
     and its respective predicted coordinates, we calculate the spherical distance between each of the spherical
     coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
     coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
     groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
     least cost in this distance matrix.

    :param gt_list: list of ground-truth Cartesian coordinates
    :param pred_list: list of predicted Cartesian coordinates
    :return: cost -  distance
    :return: less - number of DOA's missed
    :return: extra - number of DOA's over-estimated
    """

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    # Slow implementation
    # cost_mat = np.zeros((gt_len, pred_len))
    # for gt_cnt, gt in enumerate(gt_list_rad):
    #     for pred_cnt, pred in enumerate(pred_list_rad):
    #         cost_mat[gt_cnt, pred_cnt] = distance_between_spherical_coordinates_rad(gt, pred)

    # Fast implementation
    if gt_len and pred_len:
        x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], \
                               pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
        cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind].sum()
    return cost


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def sph2cart(azimuth, elevation, r):
    '''
    Convert spherical to cartesian coordinates

    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    '''

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def cart2sph(x, y, z):
    '''
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    '''

    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


###############################################################
# SELD scoring functions
###############################################################


def early_stopping_metric(sed_error, doa_error):
    """
    Compute early stopping metric from sed and doa errors.

    :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
    :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
    :return: seld metric result
    """
    seld_metric = np.mean([
        sed_error[0],
        1 - sed_error[1],
        doa_error[0]/180,
        1 - doa_error[1]]
        )
    return seld_metric

