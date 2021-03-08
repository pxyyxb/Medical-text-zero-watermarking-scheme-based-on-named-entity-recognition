import pandas as pd
import hashlib
import os
def encode(s):
    return ''.join([bin(ord(c)).replace('0b', '') for c in s])

def decode(s):
    wordlist = [s[i: i+16] for i in range(0, len(s), 16)]
    return ''.join([chr(i) for i in [int(b, 2) for b in wordlist]])

# 将特征进行二进制编码
def featurecode(order, name, length):
    return encode(name + str(order) + str(length))


def MakeMark(filename, order):
    df = pd.read_csv(filename, sep=' ', names=['name', 'begin', 'end', 'tag'])
    feature = ""
    for i in range(len(df)):
        Entity_order = i
        Entity_name = df['name'][i]
        #Entity_length = len(Entity_name)
        Entity_length = df['end'][i] - df['begin'][i] + 1
        #print(Entity_length)
        Entity_tag = df['tag'][i]
        feature = feature + featurecode(Entity_order, Entity_name, Entity_length)

    #混沌处理
    n = len(feature)
    K = [0 for x in range(0, n + 1)]
    feature = list(map(int,feature))
    #打印转换之后的0和1字符
    #print(feature)
    chaos = []
    u = 3.6
    K[0] = 0.5
    thresh = 0.8
    KK = [0 for x in range(0, n + 1)]
    for i in range(0, n):
        K[i + 1] = u * K[i] * (1 - K[i])
        #print(i,":",K[i + 1]),
        if K[i + 1] < thresh:
            KK[i + 1] = 0;
        else:
            KK[i + 1] = 1;
        #print(i,":",K[i],"-->",KK[i+1])
        L = KK[i+1] ^ feature[i]
        #print(L)
        chaos.append(str(L))
    #打印混沌之后的0 1 序列
    #print(chaos)
    '''
    name = '01序列.txt'
    if os.path.exists(name):
        name = '02序列.txt'
    f = open(name, 'w', encoding='UTF-8')
    f.write(''.join(chaos))
    f.close()
    '''

    #md5加密
    m = hashlib.md5()
    m.update(''.join(chaos).encode('utf-8'))
    result = m.hexdigest()
    #md5=hashlib.md5(chaos.encode('utf-8')).hexdigest()
    print(result)


    #水印存为txt文件
    file_handle=open(order, mode='w')
    file_handle.write(result)
    file_handle.close()


#获取文件夹下所有路径
def get_all_path(open_file_path):
    rootdir = open_file_path
    path_list = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        com_path = os.path.join(rootdir, list[i])
        #print(com_path)
        if os.path.isfile(com_path):
            path_list.append(com_path)
        if os.path.isdir(com_path):
            path_list.extend(get_all_path(com_path))
    #print(path_list)
    return path_list


#data-label-1
if __name__ == "__main__":
    #获取所有特征文本
    pathname = r'data-labeled-2'
    path_list = get_all_path(pathname)
    print(path_list)

    #批量处理
    path_now = 'WaterMarkResult'

    for i in range(0, len(path_list)):
        path = path_list[i].replace(pathname,path_now)
        MakeMark(path_list[i], path)


