import os
import numpy as np
import torch
import librosa
import panns_inference
from utlis import get_vad_decision,wav_to_mel_spectrogram
from model import SpeakerEncoder
from metrics import Metrics
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
from data_object import seg_object,segmentobj
from pyannote.core import Segment, Timeline, Annotation
import argparse

def get_vad_segmentation(vad_decision,vad_window_length=30):
    '''
    Used to merge the vad framewise output to segments 
    Input:
        vad_decision: vad binary output 1 for speech and 0 for noise 
        vad_window_length: used in webrtcvad for making the decision at the frame level 
    Ouput:
        return the seg obj with the timing info of the segements
    '''
    vad_obj = seg_object()
    start = 0
    total = 0
    prev_state = 0
    for i in vad_decision[1:]:
        if prev_state != i:
            if prev_state == 1:
                vad_obj.add_seg(start,start+total,1)
            prev_state=i
            start +=total 
            total = 0
        total+=vad_window_length/1000
    return vad_obj

def get_sed_segments(sed_model,wav,fs,vad_obj,sed_time=0.02):
    '''
    Vad doesn't good job if background has music and noise like clapping 
    we need to remove them before feeding them to speaker embedding 
    Input:
        sed_model: sound event detection model
        wav: audio data
        fd: sampling rate
        vad_obj: obj from vad segmentation 
        sed_time:  frame level time of sound event detection
    Ouput:
        return the  obj with the timing info of the segement retating only speech and removing music segments etc.
    '''
    assert isinstance(vad_obj,seg_object),'error in the segementation module'
    assert isinstance(wav,(np.ndarray, np.generic)),'wav should be of numpy'
    assert isinstance(fs,int),'sampling rate should be int'
    sed_obj = seg_object()
    for vd_seg_no,vd_seg in vad_obj.timing.items():
        try:
            wav_segment = wav[int(vd_seg.start_time*fs):int(vd_seg.end_time*fs)]
            sed_value = np.squeeze(np.argmax(sed_model.inference(np.expand_dims(wav_segment,axis=0)),axis=-1))
            sed_value = np.where(sed_value<6,0,sed_value)
            if np.all(sed_value==0):
                sed_obj.add_seg(vd_seg.start_time,vd_seg.end_time,sed.labels[0])
            else:
                # optimize this chunk of code 
                start = vd_seg.start_time
                prev = sed_value[0]
                total=sed_time
                for cur in sed_value[1:]:
                    if cur != prev:
                        if prev ==0:
                            sed_obj.add_seg(start,start+total,sed.labels[prev])
                        prev = cur
                        start+=total
                        total =0 
                    total+=sed_time
                else:
                    if prev ==0:
                        sed_obj.add_seg(start,start+total,sed.labels[prev])
        except:
            pass
    return sed_obj

def get_intra_boundary(embedding_model,sed_obj,wav,fs,device,threshold = 0.6,sp_time = 0.8,win_len = 160,hop_len = 80):
    '''
    Used identify the speaker change detection within a segment 
    Input:
        embedding_model: speaker embedding model used to extract the embedding
        sed_obj: segmentation object from sed method
        wav: audio data
        fs: sampling rate
        device: cuda or cpu where the data needs to sent for extracting the embedding
        threshold: Used for diffirencating the distance for speaker change
        sp_time: frame time for speaker embedding
        win_len: number of spectrogram frame given to the speaker embedding model
        hop_len: number of spectrogram frame to jump for next frame of speaker embedding

    Ouput:
        return the seg obj with the timing info of the segements with similar similarity
    '''
    embed_obj = seg_object()
    seg_no=0
    for i,j in sed_obj.timing.items():
        wav_seg = wav[int(j.start_time*fs):int(j.end_time*fs)]
        spec = wav_to_mel_spectrogram(wav_seg)
        if spec.shape[0]%win_len!=0:
            spec = np.pad(spec,((0,win_len-(spec.shape[0]%win_len)),(0,0)),mode='wrap')
        feat_in = np.stack([spec[st*hop_len:(st+1)*hop_len] for st in range(spec.shape[0]//hop_len-1)])
        embeds = embedding_model(torch.from_numpy(feat_in).to(device)).detach().cpu().numpy()
        start_time = j.start_time
        prev = [embeds[0,:]]
        total = sp_time
        for embed in embeds[1:]:
            simi = np.dot(np.mean(prev,axis=0),embed)
            # print(simi)
            if simi<threshold:
                if start_time+total<j.end_time:
                    embed_obj.add_seg(start_time,start_time+total,np.mean(prev,axis=0))
                else:
                    embed_obj.add_seg(start_time,j.end_time,np.mean(prev,axis=0))
                seg_no+=1
                start_time+=total
                total=0
                prev=[]
            prev.append(embed)
            total+=sp_time
        else:
            embed_obj.add_seg(start_time,j.end_time,np.mean(prev,axis=0))
            seg_no+=1
    return embed_obj

def get_inter_boundary(embed_obj,threshold=0.62):
    '''
    Used to merge similar speaker across the groups
    Input:
        embed_obj: segements with speaker based segments 
        threshold: Threshold for identifying different speaker cluster 
    Ouput:
        return the seg obj with the timing info of the segements 
    '''
    com_obj = seg_object()
    prev=None
    seg_no=0
    for i,j in embed_obj.timing.items():
        if prev is None:
            prev = j
            continue
        simi = np.dot(prev.obj,j.obj)
        if simi<threshold:
            com_obj.add_seg(prev.start_time,prev.end_time,seg_no)
            seg_no+=1
            prev = j
        else:

            prev.end_time=j.end_time
            prev.obj = np.mean(np.stack([prev.obj,j.obj]),axis=0)
    else:
        com_obj.add_seg(prev.start_time,prev.end_time,seg_no)
    return com_obj

def get_boundary(filename,device):
    '''
    Used to boundaries of the speaker change 
    Input:
        filename: audio filename 
        device: cuda or cpu 
    Ouput:
        return the seg obj with the timing info of the segements
    '''
    wav,fs = librosa.load(filename,sr=None)
    vad_decision = get_vad_decision(np.pad(wav,(0,0),constant_values=(0,0)))
    vad_segments = get_vad_segmentation(vad_decision)
    sed_segments = get_sed_segments(sed,wav,fs,vad_segments)
    intra_segments = get_intra_boundary(model,sed_segments,wav,fs,device)
    return get_inter_boundary(intra_segments)



parser = argparse.ArgumentParser()
parser.add_argument('filename',help='audio file for which speaker detection is required')
parser.add_argument('reference_path',help='reference file with the speaker segmentation infomation is present')
parser.add_argument('--speaker_embedding_model_path',default="../pretrained.pt",help='speaker embedding model used for extracting the speaker embedding')
parser.add_argument('--sed_model_path',default='/home/sumukh/panns_data/Cnn14_DecisionLevelMax.pth',help='Sound event detection model')
args = parser.parse_args()

assert os.path.isfile(args.filename),'input audio file path is incorrect'
assert os.path.isfile(args.reference_path),'reference file path is incorrect'
assert os.path.isfile(args.speaker_embedding_model_path),'speaker embedding model is missing'
assert os.path.isfile(args.sed_model_path),'sed model is not in proper path'

# sound event detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sed = SoundEventDetection(checkpoint_path=args.sed_model_path, device=device)

# speaker enconder 
model = SpeakerEncoder(device,'cpu')
checkpoint = torch.load(args.speaker_embedding_model_path,map_location=device)

assert 'model_state' in checkpoint,'model state dict is not present'
model.load_state_dict(checkpoint['model_state'])

metric = Metrics()

filename = args.filename#'/home/sumukh/speakerchange/celeb/conversation/voxconverse_test_wav/bvqnu.wav'
reference_path = args.reference_path#'/home/sumukh/speakerchange/celeb/conversation/voxconverse/test/bvqnu.txt'
reference = Annotation()
with open(reference_path,'r') as ftp:
    for line in ftp.read().split("\n")[:-1]:
        st,et,sp= line.split()
        reference[Segment(float(st),float(et))]=sp

com_obj = get_boundary(filename,device) 
hypothesis = Annotation()
for i,j in com_obj.timing.items():
    hypothesis[Segment(j.start_time,j.end_time)]=j.obj
    
print('Cluster Purity:{}'.format(metric.get_purity(reference,hypothesis)))
print('Cluster Coverage:{}'.format(metric.get_coverage(reference,hypothesis)))
print('Cluster Fmeasure:{}'.format(metric.get_overall(reference,hypothesis)))
print('Precision:{}'.format(metric.get_precision(reference,hypothesis)))
print('Recall:{}'.format(metric.get_recall(reference,hypothesis)))

