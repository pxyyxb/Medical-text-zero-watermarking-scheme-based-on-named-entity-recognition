from  watermark import MakeMark
from lstm_predict import LSTMNER

def checkdiff(mark_a, mark_b):
    if mark_a == mark_b:
        print('水印相同，文本有效。')
    else:
        print('文本被篡改！！！')

def collect_bios(filelist):
    ner = LSTMNER()
    result_paths = []
    for file in filelist:
        f = open(file, 'r', encoding='UTF-8')
        words = ''.join(f.readlines()).replace('\n','').replace('\r','')
        f.close()
        result_words, result_tags = ner.predict(words)
        result = ner.collect_entities_bio(result_words, result_tags)
        leng = 1
        bios = []
        for bio in result:
            if bio[-1] != 'o':
                bios.append(bio[:-2]+' '+str(leng)+' '+str(leng+len(bio)-3)+' '+bio[-1])
            leng = leng + len(bio) - 2
        print(bios)
        bios = '\n'.join(bios)
        bios = bios.replace('C', '检查与检验').replace('S', '症状和体征').replace('D', '疾病和诊断').replace('T', '治疗').replace('B', '身体部位')
        result_path = file.replace('.txt','-result.txt').replace('Check','Check\\实体提取结果')
        result_paths.append(result_path)
        f_result = open(result_path, 'w', encoding='UTF-8')
        f_result.write(bios)
        f_result.close()
    return result_paths

def Mark_check(real, fake):
    real_mark = real.replace('.txt', '-mark.txt').replace('Check','Check\\水印文件')
    fake_mark = []#水印文件名称
    files = [real]#文件名的名称
    for fakefile in fake:
        files.append(fakefile)
        fake_mark.append(fakefile.replace('.txt', '-mark.txt').replace('Check','Check\\水印文件'))

    bioses = collect_bios(files)#生成实体

    MakeMark(bioses[0], real_mark)
    f1 = open(real_mark, 'r').read()
    for i in range(len(fake)):
        MakeMark(bioses[i+1], fake_mark[i])#生成水印
        f2 = open(fake_mark[i], 'r').read()
        if f1 == f2:
            print(fake[i] + '与' + real + '水印相同，文本有效。')
        else:
            print(fake[i] + '与' + real + '水印不同，文本已被篡改！！！')

if __name__ == "__main__":
    real = 'Check\\病史特点-11.txt'
    fake = ['Check\\病史特点-11-无效修改.txt', 'Check\\病史特点-11-实体移位.txt',
            'Check\\病史特点-11-添加实体.txt', 'Check\\病史特点-11-替换实体.txt',
            'Check\\病史特点-11-删除实体.txt']#, 'Check\\病史特点-11-修改实体.txt']
    Mark_check(real, fake)