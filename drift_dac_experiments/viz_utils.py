# -*- coding: utf-8 -*-


def isfloat(str_a):
    try:
        float(str_a)
    except ValueError:
        return False
    return True


def name2type(shift_name):
    name_parts = shift_name.split('_')
    shift_type = '_'.join([part for part in name_parts if not isfloat(part)])
    return shift_type


def name2severity(shift_name):
    name_parts = shift_name.split('_')
    shift_params = [float(part) for part in name_parts if isfloat(part)]
    return shift_params[0]

