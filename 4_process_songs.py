'''the idea is that since librosa can directly extract mfcc features from mp3 files by converting it into a temporary wav file in the cache, take 10 seconds increment of the songs from your dataset and convert them into features, save/write those features to storage as images; which we can load for training later'''


import librosa
import math
import json
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


SAMPLE_RATE = 22050
# each segment of the song is 10 seconds long, from which a spectogram would be created
SEGMENT_DURATION = 10
# DURATION = 30 #measuring duration of song in seconds as per dataset
# SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(file,n_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
    #dictionary to store data
    data={
        "mfcc":[]
    }
    num_samples_per_segment=int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment=math.ceil(num_samples_per_segment/hop_length)

    signal,sr=librosa.load(file,sr=SAMPLE_RATE)

    #process segments extracting mfcc and storing mfcc value
    for s in range(num_segments):
        start_sample=num_samples_per_segment*s
        finish_sample=start_sample+num_samples_per_segment

        mfcc=librosa.feature.mfcc(signal[start_sample:finish_sample],sr=sr,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)

        mfcc=mfcc.T

        #store mfcc for segment if it has the expected length
        if len(mfcc)==expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
    

def save_melspectogram(song_file_path,song_fname,genre):
    ''' 
    the idea is 
    STEP 1) get the duration of the song(in seconds), 
    STEP 2) doing that would help us to know  how many 10 seconds segements we can extract from the song
    STEP 3) loop through the song in 10 second increments by providing librosa the offset and the duration of the song it should read at a particular iteration
    STEP 4) Create and write the mel spectograms of the 10 seconds snippets of the songs to disk
    '''
    song,sr = librosa.load(song_file_path,sr=SAMPLE_RATE)
    duration  = librosa.get_duration(y = song,sr = sr)
 
    # calculating how many 'n' seconds segments does this song have? 
    # in this case n =  SEGMENT_DURATION = 10
    num_segments = int(duration/SEGMENT_DURATION)

    for i in tqdm(range(num_segments)):
        # print(SEGMENT_DURATION,i*10,duration)
        y,sr = librosa.load(
            song_file_path,
            sr = SAMPLE_RATE,
            offset = i*SEGMENT_DURATION, #start reading after 0,10,20 seconds
            duration=SEGMENT_DURATION # read the next 10 sconds from the offset only
            
            )
        # saves these images with whitespaces and both the axis present
        # mels = librosa.feature.melspectrogram(y=y,sr=sr)
        # fig = plt.Figure()
        # canvas = FigureCanvas(fig)
        # p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
        # os.makedirs(f'{spectogram_save_path}/{genre}',exist_ok=True)
        # plt.savefig(f'{spectogram_save_path}/{genre}/{song_fname + "_" + str(i).zfill(3)}.png')

        # saves the images with axis,whitespaces not present. This is the one that we want, because the axis and whitespaces could be thought of as potential noise during the training
        mels = librosa.feature.melspectrogram(y=y,sr=sr)
        fig=plt.figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        os.makedirs(f'{spectogram_save_path}/{genre}',exist_ok=True)
        plt.savefig(f'{spectogram_save_path}/{genre}/{song_fname + "_" + str(i).zfill(3)}.png', bbox_inches = 'tight',pad_inches = 0)
    
                


if __name__=="__main__":
    mp3_dir_path = 'data/sample_download/mp3_download_top75'
    # spectogram_save_path = 'data/sample_download/mel_spectograms_iter_02_top5'
    spectogram_save_path = 'data/sample_download/mel_spectograms_iter_03_top75'
    
    os.makedirs(spectogram_save_path,exist_ok=True)

    for genre in os.listdir(mp3_dir_path):

        genre_dir = os.path.join(mp3_dir_path,genre)
        print("#"*50,genre_dir,"#"*50,)
            
        for song_fname in os.listdir(genre_dir):
            song_file_path = os.path.join(genre_dir,song_fname)
            print(song_fname)
            # there is a difference between using mfcc and melspecotgram as features because they both as slightly different from each other. google the difference
            # save_mfcc(
            #     song_file_path
            #     )
            save_melspectogram(
                song_file_path,
                song_fname = song_fname, # name of the song file to make a save path for the resulting images
                genre = genre, # basically the genre of the song to make a save path for the resulting images

            )

    # save_mfcc(file,JSON_PATH,num_segments=10)