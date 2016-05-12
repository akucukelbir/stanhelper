"""stanhelper.py

This module implements functions that help interface with cmdStan.

    stan_read_csv(fname)        -- Alp Kucukelbir
    stan_rdump(data, fname)     -- Allen Riddell (from PyStan)

"""

import numpy as np
import pandas as pd
from operator import mul
import linecache

import sys
import re
from numbers import Number
if sys.version_info[0] == 2:
    from collections import OrderedDict, Sequence
    string_types = (str,)
else:
    from collections.abc import OrderedDict, Sequence
    string_types = (str, unicode)

def stan_read_csv(fname):
    """Reads and parses the output file (csv) from cmdStan.

    Alp Kucukelbir (2016)

    Args:
        fname (str): the file name. 

    Returns:
        dict: a dictionary with reshaped parameter estimates and samples.
        
            If Stan `method = sampling`, then dict contains keys that match 
            the parameter names. For each parameter, its posterior samples 
            are reshaped as `(num_samples, dim 1, dim 2, ...)`.
            
            If Stan `method = optimize`, then dict contains keys that match
            the parameter names. For each parameter, its posterior mode
            is reshaped as `(dim 1, dim 2, ...)`.
            
            If Stan `method = variational`, then dict contains two dictionaries.
                dict['sampled_pars'] contains the samples drawn from the
                approximate posterior; it has the same structure as the
                `sampling` output.
            
                dict['mean_pars'] contains the means of the approximate 
                posterior; it has the same structure as the `optimize` output.

    Raises:
        RuntimeError: If the file does not contain a valid method field in
            its header (the comments in the csv file).
        RuntimeError: If the file does not have the correct number of rows.
        
    """
        
    # figure out which Stan method was used to generate csv file
    linecache.clearcache()
    method_line = linecache.getline(fname, 5)
    SAMPLE = 0
    OPTIMIZE = 1
    VARIATIONAL = 2
    
    if method_line.find('sample') != -1:
        method = SAMPLE
    elif method_line.find('optimize') != -1: 
        method = OPTIMIZE
    elif method_line.find('variational') != -1:
        method = VARIATIONAL
    else:
        raise RuntimeError(fname +' does not have a valid method field.')
    
    # read in csv file
    df = pd.read_csv(fname, comment='#')
    

    if method == SAMPLE:
        if df.shape[0] < 1:
            raise RuntimeError(fname + ' must contain at least 1 row.')

    if method == OPTIMIZE:
        if df.shape[0] != 1:
            raise RuntimeError(fname + ' must contain exactly 1 row.')
            
    if method == VARIATIONAL:
        del df['lp__']
        if df.shape[0] < 2:
            raise RuntimeError(fname + ' must contain at least 2 rows.')
    
    # get parameter names and dimensions
    column_names_split = df.columns.map(lambda x: x.split('.'))
    par_names = OrderedDict()
    for entry in column_names_split:
        if len(entry[1:]) != 0:
            par_names[entry[0]] = entry[1:]
        else:
            par_names[entry[0]] = [1]
            
    # get first line of parameters
    first_line = df.ix[0].values
    
    first_line_pars = OrderedDict()
    ofs = 0
    for par in par_names:
        shape = tuple(map(int, par_names[par]))
        jump_ahead = reduce(mul,shape)
        if len(shape) != 1:
            first_line_pars[par] = np.reshape(first_line[ofs:ofs+jump_ahead], 
                                              shape, order='F')
        else:
            first_line_pars[par] = first_line[ofs:ofs+jump_ahead]
        ofs = ofs + jump_ahead
    
    # RETURN OPTIMIZE: the first line contains the MAP estimates.
    if method == OPTIMIZE:
        return first_line_pars
    
    # get number of samples
    if method == VARIATIONAL:
        num_samples = df.shape[0]-1
    if method == SAMPLE:
        num_samples = df.shape[0]
    
    # get samples
    sampled_pars = OrderedDict()
    ofs = 0
    for par in par_names:
        shape_list = map(int, par_names[par])
        jump_ahead = reduce(mul,shape_list)
        shape_list.insert(0,num_samples)
        shape = tuple(shape_list)
        if len(shape) != 2:
            sampled_pars[par] = np.reshape(
                                df.tail(num_samples).values[:,ofs:ofs+jump_ahead], 
                                shape, order='F')
        else:
            sampled_pars[par] = df.tail(num_samples).values[:,ofs:ofs+jump_ahead]
        ofs = ofs + jump_ahead
    
    result = dict()
    # RETURN VARIATIONAL: the first line contains the posterior mean estimates.
    if method == VARIATIONAL:
        result['mean_pars'] = first_line_pars
        result['sampled_pars'] = sampled_pars
        return result
    
    # RETURN SAMPLE
    if method == SAMPLE:
        result = sampled_pars
        return result

def is_legal_stan_vname(name):
    stan_kw1 = ('for', 'in', 'while', 'repeat', 'until', 'if', 'then', 'else',
                'true', 'false')
    stan_kw2 = ('int', 'real', 'vector', 'simplex', 'ordered', 'positive_ordered',
                'row_vector', 'matrix', 'corr_matrix', 'cov_matrix', 'lower', 'upper')
    stan_kw3 = ('model', 'data', 'parameters', 'quantities', 'transformed', 'generated')
    cpp_kw = ("alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool",
              "break", "case", "catch", "char", "char16_t", "char32_t", "class", "compl",
              "const", "constexpr", "const_cast", "continue", "decltype", "default", "delete",
              "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern",
              "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable",
              "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
              "private", "protected", "public", "register", "reinterpret_cast", "return",
              "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
              "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef",
              "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile",
              "wchar_t", "while", "xor", "xor_eq")
    illegal = stan_kw1 + stan_kw2 + stan_kw3 + cpp_kw
    if re.findall(r'(\.|^[0-9]|__$)', name):
        return False
    return not name in illegal

def _dict_to_rdump(data):
    parts = []
    for name, value in data.items():
        if isinstance(value, (Sequence, Number, np.number, np.ndarray, int, bool, float)) \
           and not isinstance(value, string_types):
            value = np.asarray(value)
        else:
            raise ValueError("Variable {} is not a number and cannot be dumped.".format(name))

        if value.dtype == np.bool:
            value = value.astype(int)

        if value.ndim == 0:
            s = '{} <- {}\n'.format(name, str(value))
        elif value.ndim == 1:
            if len(value) == 1:
                s = '{} <- {}\n'.format(name, str(value.pop()))
            else:
                s = '{} <-\nc({})\n'.format(name, ', '.join(str(v) for v in value))
        elif value.ndim > 1:
            tmpl = '{} <-\nstructure(c({}), .Dim = c({}))\n'
            # transpose value as R uses column-major
            # 'F' = Fortran, column-major
            s = tmpl.format(name,
                            ', '.join(str(v) for v in value.flatten(order='F')),
                            ', '.join(str(v) for v in value.shape))
        parts.append(s)
    return ''.join(parts)

def _rdump_value_to_numpy(s):
    """
    Convert a R dump formatted value to Numpy equivalent
    For example, "c(1, 2)" becomes ``array([1, 2])``
    Only supports a few R data structures. Will not work with European decimal format.
    """
    if "structure" in s:
        vector_str, shape_str = re.findall(r'c\([^\)]+\)', s)
        shape = [int(d) for d in shape_str[2:-1].split(',')]
        if '.' in vector_str:
            arr = np.array([float(v) for v in vector_str[2:-1].split(',')])
        else:
            arr = np.array([int(v) for v in vector_str[2:-1].split(',')])
        # 'F' = Fortran, column-major
        arr = arr.reshape(shape, order='F')
    elif "c(" in s:
        if '.' in s:
            arr = np.array([float(v) for v in s[2:-1].split(',')], order='F')
        else:
            arr = np.array([int(v) for v in s[2:-1].split(',')], order='F')
    else:
        arr = np.array(float(s) if '.' in s else int(s))
    return arr

def stan_rdump(data, filename):
    """
    Dump a dictionary with model data into a file using the R dump format that
    Stan supports.

    Parameters
    ----------
    data : dict
    filename : str

    """
    for name in data:
        if not is_legal_stan_vname(name):
            raise ValueError("Variable name {} is not allowed in Stan".format(name))
    with open(filename, 'w') as f:
        f.write(_dict_to_rdump(data))

def read_rdump(filename):
    """
    Read data formatted using the R dump format
    Parameters
    ----------
    filename: str
    Returns
    -------
    data : OrderedDict
    """
    contents = open(filename).read().strip()
    names = [name.strip() for name in re.findall(r'^(\w+) <-', contents, re.MULTILINE)]
    values = [value.strip() for value in re.split('\w+ +<-', contents) if value]
    if len(values) != len(names):
        raise ValueError("Unable to read file. Unable to pair variable name with value.")
    d = OrderedDict()
    for name, value in zip(names, values):
        d[name.strip()] = _rdump_value_to_numpy(value.strip())
    return d
