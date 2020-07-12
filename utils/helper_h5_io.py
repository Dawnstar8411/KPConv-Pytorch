# ！/usr/bin/python3
# -*- coding: utf-8 -*-

import h5py


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return data, label, seg


def load_h5_data_label_nomal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return data, label, normal


def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()


def save_h5_data_label_normal(h5_filename, data, label, normal, data_dtype='float32', label_dtype='unit8',
                              normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('normal', data=normal, compression='gzip', compression_opts=4, dtype=normal_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()
