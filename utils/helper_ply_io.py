# ！/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import sys

# 设置一个字典
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# 读取模式，按ascii模式读取
# 小端读取：低字节对应起始地址（低位内存）
# 大端读取：高字节对应起始地址（低位内存）
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}


def parse_header(plyfile, ext):
    line = []
    properties = []
    num_points = None
    # 只读取文件头
    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            # elif current_element == 'face':
            #    if not line.startswith('property list uchar int'):
            #        raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def header_properties(field_list, field_names):
    # 保存需要写入的数据
    lines = []

    # 一共有多少个点
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def describe_element(name, df):
    """
    Takes the columns of the dataframe and builds a ply-like description
    :param name: str
    :param df: pandas DataFrame
    :return: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):  # df.columns, 是一个list, df中每一列的表头名称
            # 得到数据类型的第一个字母，然后去字典查询
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    :param filename: string, the name of the file to which the data is saved.A '.ply' extension will be appended to the
        file name if it does no already have one.
    :param field_list:  (list/tuple/numpy)array, the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.
    :param field_names: list, the name of each fields as a list of strings. Has to be the same length as the number of
        fields.
    :param triangular_faces:
    :return:

    Examples
    >>> points = np.random.rand(10,3)
    >>> write_ply('example1.ply',points,['x','y','z'])

    >>> values = np.random.randint(2,size=10)
    >>> write_ply('example2.ply',[points,values],['x','y','z','values'])

    >>> colors = np.random.randint(255,size=(10,3),dtype=np.uint8)
    >>> field_names = ['x','y','z','red','green','blue','values']
    >>> write_ply('example3.ply',[points, colors, values],field_names)
    """
    # 全都转换成list,list中的每个元素为一个numpy数组，数组可以有1列或多列
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):  # 对于list中的每个数组
        if field.ndim < 2:  # 如果是一维数组，扩展成n行1列的二维数组
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

    # 输入list中的每一个numpy数组应该具有相同的行数，意味着有相同的点
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

    # 输入list中的所有numpy数组的列数加起来应该等于输入name列表的元素个数，意味着每个name对应一列
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # 可以自动为文件添加后缀
    if not filename.endswith('.ply'):
        filename += '.ply'

    # 打开文件
    with open(filename, 'w') as plyfile:

        # 第一个词
        header = ['ply']

        # 系统编码格式，大端还是小端，从系统设置中读出
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # 数据的属性
        header.extend(header_properties(field_list, field_names))

        # 添加三角网格的相关参数
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # 文件头结束
        header.append('end_header')

        # 将文件头逐行写入文件
        for line in header:
            plyfile.write("%s\n" % line)

    # 打开文件，a 附加写，不覆盖之前的，b 二进制方式写入
    with open(filename, 'ab') as plyfile:

        i = 0
        type_list = []
        for fields in field_list:  # 对每一个numpy，n行若干列
            for field in fields.T:  # 数组转置后的每一行，
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field  # 读取数据保存到data中，data是一个字典
                i += 1

        data.tofile(plyfile)

        # 写入三角面片的数据
        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def read_ply(filename, triangular_mesh=False):
    """
    :param filename: string, the name of the file to read
    :param triangular_mesh:
    :return: array, data stored in the file
    Examples
    Store data in file
    >>> points = np.random.rand(5,3)
    >>> values = np.random.randint(2,size=10)
    >>> write_ply('example.ply',[points,values],['x','y','z','values'])

    Read the file
    >>> data = read_ply('example.ply')
    >>> vales = data['values']
    array([0,0,1,1,0])

    >>> points = np.vstack((data['x'],data['y'],data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """
    # 打开文件，二进制类型读
    with open(filename, 'rb') as plyfile:

        # 判断文件是否以'ply开头'
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start with the word ply')

        # 获取文件数据的编码方式
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # 根据读写方式去字典查询特定符号
        ext = valid_formats[fmt]
        # 如果要读取三角面片的内容
        if triangular_mesh:
            # 读取文件头，得到点的数量，面片的数量和数据类型
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)
            # 读取点云数据
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)
            # 三角面片的数据类型
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            # 读取三角面片数据
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)
            # 组合三角面片坐标形成数组并转置
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T

            data = [vertex_data, faces]

        else:
            # 从文件头中获取点的数量和数据类型
            num_points, properties = parse_header(plyfile, ext)
            # 读取数据，保存成numpy数组
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data

