import os
from tokenize import TokenError

def txtfromrtm(inputPath,outputPath):
    '''
    used to convert rtmm file to txt based format
    
    Input:
        inputPath: path where the rtmm file is kept
        outputPath: path where the txt file is stored
    '''
    assert os.path.isfile(inputPath),'rtmm file path is incorrect'
    label_ftp = open(outputPath,'w')
    with open(inputPath,'r') as ftp:
        prev = None
        for line in ftp.read().split("\n")[:-1]:
            line = line.split()
            #print(line[3],line[4],line[7])
            if prev is None:
                prev = [0,float(line[4])+float(line[3]),line[7]]
                continue
            if prev[-1] != line[7]:
                #print(prev,line)
                label_ftp.write('{0}\t{1}\t{2}\n'.format(prev[0],prev[1],prev[2]))
                prev = [float(prev[1]),float(line[4])+float(line[3]),line[7]]
            else:
                prev[1] = float(line[4])+float(line[3])
        else:
            label_ftp.write('{0}\t{1}\t{2}\n'.format(prev[0],prev[1],prev[2]))
    label_ftp.close()


src_path = '/home/sumukh/speakerchange/celeb/conversation/voxconverse/dev'
for root,sub_folder,files in os.walk(src_path):
    for file in files:
        if os.path.splitext(file)[-1]==".rttm":
            input_p = os.path.join(root,file)
            ouput_p= os.path.join(root,file.split('.')[0]+".txt")
            print(input_p,ouput_p)
            txtfromrtm(input_p,ouput_p)