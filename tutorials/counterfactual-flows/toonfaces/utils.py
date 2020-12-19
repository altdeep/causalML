import os
from os.path import isfile, isdir
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from functools import reduce
import operator
from IPython.display import Image, display
import pickle

# create a template ordered dict 
baseline_attribute_dict = OrderedDict(
    dict(eye_angle=0,
         eye_lashes=0,
         eye_lid=0,
         chin_length=0,
         eyebrow_weight=0,
         eyebrow_shape=0,
         eyebrow_thickness=0,
         face_shape=0,
         facial_hair=0,
         hair=0,
         eye_color=0,
         face_color=0,
         hair_color=0,
         glasses=0,
         glasses_color=0,
         eye_slant=0,
         eyebrow_width=0,
         eye_eyebrow_distance=0
        )
)


def flatten(xs):
    return reduce(operator.iconcat, xs, [])

def negate(pred_fn):
    return (lambda x: not pred_fn(x))

def list2dirs(directory, size=None):
    def list_at(p):
        abs_path = os.path.abspath(os.path.join(directory, p))
        return [os.path.join(abs_path, f) for f in os.listdir(abs_path)]
    top_level_all = list_at(".")
    top_level_dirs = filter(isdir, top_level_all)
    top_level_files = filter(negate(isdir), top_level_all)
    snd_level_all = list(flatten(map(list_at, top_level_dirs)))
    all_files = list(top_level_files) + snd_level_all
    return all_files if size is not None else all_files[:size]

def find2dir(directory, filename):
    top_level = os.listdir(directory)
    make_path = lambda d: os.path.join(d, filename)

    if isfile(make_path(directory)):
        return make_path(directory)
    else:
        ps = list(filter(isfile, map(make_path, filter(isdir, top_level))))
        return None if len(ps) == 0 else ps[0]

def get_image_attribute_dict(directory, attribute_file_type='.csv', 
                             attribute_index_col=0, attribute_val_col=0, depth=1, size=None):
    '''
    Create dictionary of (img_filename, attribute_dict) key-value pairs.
    
    :param directory: Union[str, List[str]], path to folder or list of folders where images and their attribute files are stored
    :param attribute_file_type: str, file type of files that contain the 
        attribute values. Default is '.csv'
    :param attribute_index_col: int, column within an attribute file that pandas 
        should treat as the index column.
    :param attribute_val_col: int, column within the attribute pandas dataframe 
        that will contain the attribute values. Note: if attribute_index_col is 
        not None, the column in the attribute file and the attribute pandas 
        dataframe will not be the same. 
    :return img_attribute_dict: dict of OrderedDict, (key, value) pairs are 
        (image_name, attribute_dict) pairs, where attribute_dict is an 
        OrderedDict that has the same template ordering as 
        baseline_attribute_dict. 
    '''
    file_names = list2dirs(directory, size=size)
    attribute_file_names = [f for f in file_names if f.endswith(attribute_file_type)]
    img_attribute_dict = dict()
    for f in tqdm(attribute_file_names):
        key = f.split('.')[0]
        values_df = pd.read_csv(os.path.join(directory, f), 
            index_col=attribute_index_col, 
            header=None).iloc[:, attribute_val_col]
        assert set(values_df.index) == set(baseline_attribute_dict.keys())
        temp = values_df.to_dict()
        ordered_dict = baseline_attribute_dict.copy()
        for k, v in temp.items():
            ordered_dict[k] = v
        img_attribute_dict[key] = ordered_dict 
    return img_attribute_dict

def get_attribute_to_file_dict(att_dict):
    att_to_file_dict = dict()
    for img_name, v in tqdm(att_dict.items()):
        att_str = ''.join([str(att_val).zfill(3) for att_val in v.values()])
        att_to_file_dict[att_str] = img_name
    return att_to_file_dict

def attributes_to_key(ordered_att_dict):
    '''
    Given a dict of attribute (attribute, att_value) pairs, create a key string.
    
    :param ordered_att_dict: OrderedDict, (attribute, attribute_value) pairs 
        for images.
    :return key: str, a string representation of the attribute_value s, 
        formatted in a uniform way based on ordering of attributes in the 
        OrderedDict.
    '''
    key = ''.join([str(att_val).zfill(3) for att_val in ordered_att_dict.values()])
    return key

def key_to_attributes(key):
    '''
    Given a key representing attribute values, create a dictionary of attribute 
    (attribute, attribute_value) pairs.
    
    :param key: str, a string representation of the attribute_value s, formatted
        in a uniform way based on ordering of attributes in the OrderedDict.
    :return ordered_att_dict: OrderedDict, (attribute, attribute_value) pairs 
        for images.
    '''
    num_att = int(len(key) / 3)
    dict_keys = list(baseline_attribute_dict.keys())
    ordered_att_dict = baseline_attribute_dict.copy()
    for i in range(num_att):
        val = int(key[i*3:(i+1)*3])
        ordered_att_dict[dict_keys[i]] = val
    return ordered_att_dict

def render_img_from_attributes(attribute_dict, att_to_file_dict, 
                               img_dir='cartoonset10k', img_type='.png', display_img=False):
    attribute_key = attributes_to_key(attribute_dict)
    img_name = att_to_file_dict[attribute_key] + img_type
    if img_dir is not None:
        img_name = find2dir(img_dir, img_name)
    img = Image(filename=img_name)
    if display_img:
        display(img)
    return img


def img_from_attributes(attribute_dict, att_to_file_dict, img_dir='cartoonset10k', img_type='.png'):
    attribute_key = attributes_to_key(attribute_dict)
    img_name = att_to_file_dict[attribute_key] + img_type
    if img_dir is not None:
        img_name = find2dir(img_dir, img_name)
    img = Image(filename=img_name)
    return img


def load_from_path(dataset, size=None):
    attr_dict = get_image_attribute_dict(dataset, size=size)
    return get_attribute_to_file_dict(attr_dict)

def save_dict(attrdict, path):
    with open(path, 'wb') as p:
        pickle.dump(attrdict, p)

def load_from_pickle(path):
    with open(path, 'rb') as p:
        return pickle.load(p)

def get_csv_attrs(path):
    return pd.read_csv(path, index_col=0, header=None).iloc[:,0].to_dict()
