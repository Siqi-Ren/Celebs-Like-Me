import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
from tqdm.notebook import tqdm
from urllib.request import urlopen, urlretrieve, Request
from joblib import Parallel, delayed
from mtcnn import MTCNN
from skimage import io
detector = MTCNN()

def download_img(i,r):
    commentls=[]
    name = r["name"]
    url = r["imgurl"]
    no = r["no"]
    region = r["region"]
    dir_path=f"../data/{region}"
    if os.path.isfile(os.path.join(dir_path, name + "_" + str(no) + ".jpg")):
        pass
    else:
        imgname = None
        save_path = None
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            arr = np.asarray(bytearray(urlopen(req, timeout = 5).read()), dtype=np.uint8)
                #img = cv2.imdecode(arr, -1) ##Keep Alpha Channel
            img = cv2.imdecode(arr, 1)  ## Convert to 3 Channel
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detected = detector.detect_faces(image)
            if len(detected)==1:
                x,y,w,h = detected[0]["box"]
                x=max(x,0)
                y=max(y,0)
                w=max(w,0)
                h=max(h,0)
                crop = image[y:y+h,x:x+w]
                im = Image.fromarray(crop) 
                img = cv2.resize(np.array(im), (224,224))
                imgname = name + "_" + str(no) + ".jpg"
                save_path = os.path.join(dir_path, imgname)
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                comment= "single face found and saved"
            elif len(detected)>1:
                comment= "multiple face found"
            else:
                comment= "no face found"
            commentls.append(comment)
        except Exception as e:
            print(e, url)
            comment= "error"
            commentls.append(comment)
        return [region,name,no,url,imgname,commentls,save_path]
    
def _get_success_fail(df,col_check, cols_count = ['imgurl']):
    df_success = df[~df[col_check].isnull()]
    df_fail = df[df[col_check].isnull()]
    success_count = max([len(df_success[col_count].unique()) for col_count in cols_count if col_count in df])
    fail_count = max([len(df_fail[col_count].unique()) for col_count in cols_count if col_count in df])
    print('Success images: ', success_count)
    print('Fail images: ', fail_count)
    return df_success, df_fail

def download_retry(df,n_jobs = 1):
    data = Parallel(n_jobs=int(n_jobs))(
        delayed(download_img)(i,r) for (i, r) in tqdm(df.iterrows(), total=len(df)))
    download_cols= ["region","name","no","imgurl","imgname","comments","save_path"]
    df_download = pd.DataFrame(data,columns=download_cols)
    df_success, df_fail = _get_success_fail(df_download, col_check = 'imgname')
    df_download = pd.DataFrame()
    while len(df_success) > 0 and len(df_fail) > 0:
        df_download = pd.concat([df_download, df_success])
        print()
        print('Retrying...')
        data = Parallel(n_jobs=int(n_jobs))(
                delayed(download_img)(i,r) \
                for (i, r) in tqdm(df_fail.iterrows(), total=len(df_fail)))
        df_download_retry = pd.DataFrame(data, columns=download_cols)
        df_success, df_fail = _get_success_fail(df_download_retry, col_check = 'imgname')
        if len(df_success) == 0:
            df_download = pd.concat([df_download, df_success])
            print()
            print('Retrying with 1 thread...')
            data = Parallel(n_jobs=int(1))(
                delayed(download_img)(i,r) \
                for (i, r) in tqdm(df_fail.iterrows(), total=len(df_fail)))
            df_download_retry = pd.DataFrame(data, columns=download_cols)
            df_success, df_fail = _get_success_fail(df_download_retry, col_check = 'imgname')
        blacklist+=df_fail["imgurl"].tolist()
    if len(df_fail) > 0:
        print()
        print('Example Download Fails:')
        display(df_fail.head())
    else:
        print()
        print('Succeed to download all data!')
    df_download = pd.concat([df_download, df_success])
    return df_download, df_fail