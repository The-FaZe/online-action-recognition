
import glob     # The glob module finds all the pathnames matching a specified pattern. 
import fnmatch  # This module provides support for Unix shell-style wildcards.
import os       # This module provides a portable way of using operating system dependent functionality.
import random   # This module implements pseudo-random number generators for various distributions.
from anet_db import ANetDB  # anet_db module organize the activitynet dataset


def parse_directory(path, rgb_prefix='img_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    
    :Param path: str - represent the path to the folder of extracted frames
    :Param rgb_prefix: Str - prefix of rgb frames
    :Param flow_x_prefix: Str - prefix of flow_x frames
    :Param flow_y_prefix: Str - prefix of flow_y frames
    """
    print('parse frames under folder {}'.format(path))
    
    # create list that will contain directries of all folders in the givin path.
    # note that the '*' mean join all folders_names in that directory to path 
    # frame_folders = [dir_of_Folder1, dir_of_folder2, ... ]
    frame_folders = glob.glob(os.path.join(path, '*'))

    def count_files(directory, prefix_list):
        """
        :Param directory: Str - directory to the files to be counted
        :Param prefix_list: tuple - subset prefix of folders names that we need to count frames inside like img_, flow_x_, flow_y_
        
        :Return dir_dict: dict - its keys are the folders names, and values are the directories for each forlder
        :Return rgb_counts: dict - ??
        :Return flow_counts: dict - ??
        """
        # Create a list containing the names of the entries in the directory given by path. 
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    rgb_counts = {}     # empty dict
    flow_counts = {}
    dir_dict = {}
    
    # i: counter, f: url of frames folders
    for i,f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = f.split('/')[-1]
        rgb_counts[k] = all_cnt[0]
        dir_dict[k] = f

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        flow_counts[k] = x_cnt
        if i % 200 == 0:
            print('{} videos parsed'.format(i)) 

    print('frame folder analysis done')
    return dir_dict, rgb_counts, flow_counts


def build_split_list(split_tuple, frame_info, split_idx, shuffle=False):
    split = split_tuple[split_idx]

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            frame_dir = frame_info[0][item[0]]
            rgb_cnt = frame_info[1][item[0]]
            flow_cnt = frame_info[2][item[0]]
            rgb_list.append('{} {} {}\n'.format(frame_dir, rgb_cnt, item[1]))
            flow_list.append('{} {} {}\n'.format(frame_dir, flow_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)


## Dataset specific split file parse
# =============================================================================
# def parse_ucf_splits():
#     class_ind = [x.strip().split() for x in open('data/ucf101_splits/classInd.txt')]
#     class_mapping = {x[1]:int(x[0])-1 for x in class_ind}
# 
#     def line2rec(line):
#         items = line.strip().split('/')
#         label = class_mapping[items[0]]
#         vid = items[1].split('.')[0]
#         return vid, label
# 
#     splits = []
#     for i in range(1, 4):
#         train_list = [line2rec(x) for x in open('data/ucf101_splits/trainlist{:02d}.txt'.format(i))]
#         test_list = [line2rec(x) for x in open('data/ucf101_splits/testlist{:02d}.txt'.format(i))]
#         splits.append((train_list, test_list))
#     return splits
# 
# =============================================================================

def parse_activitynet_splits(version):
    db = ANetDB.get_db(version)
    train_instance = db.get_subset_instance('training')
    val_instance = db.get_subset_instance('validation')
    test_instance = db.get_subset_videos('testing')

    splits = []

    train_list = [(x.name, x.num_label) for x in train_instance]
    val_list = [(x.name, x.num_label) for x in val_instance]
    test_list = [(x.id, 0) for x in test_instance]

    splits.append((train_list, val_list))
    splits.append((train_list + val_list, test_list))

    return splits