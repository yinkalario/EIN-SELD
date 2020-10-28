import numpy as np
import pandas as pd
import torch


def _segment_index(x, chunklen, hoplen, last_frame_always_paddding=False):
    """Segment input x with chunklen, hoplen parameters. Return

    Args:
        x: input, time domain or feature domain (channels, time)
        chunklen:
        hoplen:
        last_frame_always_paddding: to decide if always padding for the last frame
    
    Return:
        segmented_indexes: [(begin_index, end_index), (begin_index, end_index), ...]
        segmented_pad_width: [(before, after), (before, after), ...]
    """
    x_len = x.shape[1]

    segmented_indexes = []
    segmented_pad_width = []
    if x_len < chunklen:
        begin_index = 0
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - x_len
        segmented_indexes.append((begin_index, end_index))
        segmented_pad_width.append((pad_width_before, pad_width_after))
        return segmented_indexes, segmented_pad_width

    n_frames = 1 + (x_len - chunklen) // hoplen
    for n in range(n_frames):
        begin_index = n * hoplen
        end_index = n * hoplen + chunklen
        segmented_indexes.append((begin_index, end_index))
        pad_width_before = 0
        pad_width_after = 0
        segmented_pad_width.append((pad_width_before, pad_width_after))
    
    if (n_frames - 1) * hoplen + chunklen == x_len:
        return segmented_indexes, segmented_pad_width

    # the last frame
    if last_frame_always_paddding:
        begin_index = n_frames * hoplen
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - (x_len - n_frames * hoplen)        
    else:
        if x_len - n_frames * hoplen >= chunklen // 2:
            begin_index = n_frames * hoplen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = chunklen - (x_len - n_frames * hoplen)
        else:
            begin_index = x_len - chunklen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = 0
    segmented_indexes.append((begin_index, end_index))
    segmented_pad_width.append((pad_width_before, pad_width_after))

    return segmented_indexes, segmented_pad_width


def load_dcase_format(meta_path, frame_begin_index=0, frame_length=600, num_classes=14, set_type='gt'):
    """ Load meta into dcase format

    Args:
        meta_path (Path obj): path of meta file
        frame_begin_index (int): frame begin index, for concatenating labels
        frame_length (int): frame length in a file
        num_classes (int): number of classes
    Output:
        output_dict: return a dict containing dcase output format
            output_dict[frame-containing-events] = [[class_index_1, azi_1 in degree, ele_1 in degree], [class_index_2, azi_2 in degree, ele_2 in degree]]
        sed_metrics2019: (frame, num_classes)
        doa_metrics2019: (frame, 2*num_classes), with (frame, 0:num_classes) represents azimuth, (frame, num_classes:2*num_classes) represents elevation
            both are in radiance
    """
    df = pd.read_csv(meta_path, header=None)

    output_dict = {}
    sed_metrics2019 = np.zeros((frame_length, num_classes))
    doa_metrics2019 = np.zeros((frame_length, 2*num_classes))
    for row in df.iterrows():
        frame_idx = row[1][0]
        frame_idx2020 = frame_idx + frame_begin_index
        event_idx = row[1][1]
        if set_type == 'gt':
            azi = row[1][3]
            ele = row[1][4]
        elif set_type == 'pred':
            azi = row[1][2]
            ele = row[1][3]
        if frame_idx2020 not in output_dict:
            output_dict[frame_idx2020] = []
        output_dict[frame_idx2020].append([event_idx, azi, ele])
        sed_metrics2019[frame_idx, event_idx] = 1.0
        doa_metrics2019[frame_idx, event_idx], doa_metrics2019[frame_idx, event_idx + num_classes] \
            = azi * np.pi / 180.0, ele * np.pi / 180.0
    return output_dict, sed_metrics2019, doa_metrics2019


def to_metrics2020_format(label_dict, num_frames, label_resolution):
    """Collect class-wise sound event location information in segments of length 1s (according to DCASE2020) from reference dataset

    Reference:
        https://github.com/sharathadavanne/seld-dcase2020/blob/74a0e1db61cee32c19ea9dde87ba1a5389eb9a85/cls_feature_class.py#L312
    Args:
        label_dict: Dictionary containing frame-wise sound event time and location information. Dcase format.
        num_frames: Total number of frames in the recording.
        label_resolution: Groundtruth label resolution.
    Output:
        output_dict: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth in degree, elevation in degree)
    """

    num_label_frames_1s = int(1 / label_resolution)
    num_blocks = int(np.ceil(num_frames / float(num_label_frames_1s)))
    output_dict = {x: {} for x in range(num_blocks)}
    for n_frame in range(0, num_frames, num_label_frames_1s):
        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        n_block = n_frame // num_label_frames_1s
        loc_dict = {}
        for audio_frame in range(n_frame, n_frame + num_label_frames_1s):
            if audio_frame not in label_dict:
                continue            
            for value in label_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}
                
                block_frame = audio_frame - n_frame
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for n_class in loc_dict:
            if n_class not in output_dict[n_block]:
                output_dict[n_block][n_class] = []

            keys = [k for k in loc_dict[n_class]]
            values = [loc_dict[n_class][k] for k in loc_dict[n_class]]

            output_dict[n_block][n_class].append([keys, values])

    return output_dict


