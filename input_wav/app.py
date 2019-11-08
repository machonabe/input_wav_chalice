from chalice import Chalice
import boto3
from botocore.exceptions import ClientError
import base64
import ulid
import json
import wave
import numpy as np
import tempfile
import random
import matplotlib.pyplot as plt
import csv


BUCKET_NAME = ''
s3 = boto3.client('s3')

app = Chalice(app_name='input_wav')
app.debug = True

def wav2data(wf):
    buf = wf.readframes(wf.getnframes())
    data = np.frombuffer(buf, dtype="int16")
    return data


def downsampling(conversion_rate,data,fs):
    """
    ダウンサンプリングを行う．
    入力として，変換レートとデータとサンプリング周波数．
    ダウンサンプリング後のデータとサンプリング周波数を返す．
    """
    # 間引くサンプル数を決める
    decimationSampleNum = conversion_rate-1

    # FIRフィルタの用意をする
    nyqF = fs/2.0             # 変換後のナイキスト周波数
    cF = (fs/conversion_rate/2.0-500.)/nyqF     # カットオフ周波数を設定（変換前のナイキスト周波数より少し下を設定）
    taps = 511                                  # フィルタ係数（奇数じゃないとだめ）
    # b = signal.firwin(taps, cF)           # LPFを用意

    #フィルタリング
    # data = scipy.signal.lfilter(b,1,data)

    #間引き処理
    downData = []
    for i in range(0,len(data),decimationSampleNum+1):
        downData.append(data[i])

    return np.array(downData)


def voice_trimming(data):    
    data = np.frombuffer(data, dtype="int16")
    
    threshold = 5000
    data[data>threshold]
    
    voices = np.where(np.absolute(data)>threshold)
    start_index = voices[0][0]
    end_index = voices[0][len(voices[0])-1]
    trimmed_data = data[start_index:end_index]
    return trimmed_data


def data_normalize(trimmed_data, axis=None):
    min = trimmed_data.min(axis=axis, keepdims=True)
    max = trimmed_data.max(axis=axis, keepdims=True)
    norm_data = (trimmed_data-min)/(max-min)
    return norm_data
    
    
def zscore(trimmed_data, axis = None):
    xmean = trimmed_data.mean(axis=axis, keepdims=True)
    xstd  = np.std(trimmed_data, axis=axis, keepdims=True)
    zscore = (trimmed_data-xmean)/xstd
    return zscore
    

def calc_pecgram(wf, voice_type): 
    data = wav2data(wf)
    trimmed_data = voice_trimming(data)
    if voice_type == 'pokemon':
        zscore_data = zscore(trimmed_data, axis = None)
    else:
        downsampled_data = downsampling(2,trimmed_data,48000)
        zscore_data = zscore(downsampled_data, axis = None)
        

    # FFTのサンプル数 ハイパラ
    N = 8192
    
    # FFTで用いるハミング窓
    hammingWindow = np.hamming(N)

    # スペクトログラムを描画
    pxx, freqs, bins, im = plt.specgram(zscore_data, NFFT=N, Fs=len(zscore_data), noverlap=0, window=hammingWindow)
    
    return (pxx, freqs, bins, im)


def append_zero(vec):
    max_len = 4
    if len(vec) != max_len:
        append = np.zeros(max_len - len(vec))
        extended_vec = np.append(vec, append)
    else: extended_vec = vec
    return extended_vec


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def voice_digitization(wav):
  specgram_voice = calc_pecgram(wav, 'person')
  matcrix_voice = specgram_voice[0]
  digit = np.amax(matcrix_voice, axis=0)

  return append_zero(digit)

def decide_most_similar_pokemon(voice_digit):
    csv_path = '/tmp/compare.csv'
    try:
      s3.download_file(BUCKET_NAME, 'compare.csv', csv_path)
    except ClientError as e:
      print(e)

    with open(csv_path) as f:
      reader = csv.reader(f)
      pokemon_no = ""
      pokemon_name = ""
      similarity = 0
      for row in reader:
        sim = cos_sim(voice_digit, np.array(row[2:], dtype='float64'))
        # if sim >= 0.9:
        #   continue
        if similarity < sim:
          pokemon_no = row[0]
          pokemon_name = row[1]
          similarity = cos_sim(voice_digit, np.array(row[2:], dtype='float64'))
      return pokemon_no, pokemon_name, similarity

@app.route('/upload', methods=['POST'],
           content_types=['audio/wav'])
def upload():
    data = app.current_request.raw_body

    uuid = ulid.new().str
    key = 'user-voice' + '/' + uuid + '.wav'

    data = data.split(b'\r\n')  
    wavBody = data[4]

    boundary = data[0].replace(b'-', b'')

    for i in range(5, len(data)):
        if data[i].find(boundary) != -1:
            break
        else:
            wavBody = wavBody + b'\r\n' + data[i]


    try:
      s3.put_object(Bucket=BUCKET_NAME, Key=key, ContentType='audio/wav', Body=wavBody)
    except ClientError as e:
      print(e)

    return 

############
# curl -X POST https://szcac7x7jc.execute-api.ap-northeast-1.amazonaws.com/api/upload -H "Content-Type: audio/wav" -F  "file=@voice4.wav"
# 上記のcurlで送った場合にS3に正常にputされる
############