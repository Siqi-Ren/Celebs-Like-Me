import os
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import argparse
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import glob
import pandas as pd
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, TensorDataset
from tqdm.notebook import tqdm
from scipy.spatial import distance
detector = MTCNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


detector = MTCNN()

def detect_and_crop_face(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    detected = detector.detect_faces(image)
    if len(detected)==1:
        x,y,w,h = detected[0]["box"]
        x=max(x,0)
        y=max(y,0)
        w=max(w,0)
        h=max(h,0)
        crop = image[y:y+h,x:x+w]
        im = Image.fromarray(crop)
        img = np.array(im)
        img = cv2.resize(img, (224, 224))
        return img

class MyDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, img, transform=None):
        self.tensors = np.expand_dims(img, axis = 0)
        self.transform = transform

    def __getitem__(self, index):
        img = self.tensors[index]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img
    def __len__(self):
            return self.tensors.shape[0]
    
    
def load_image(img,Tencrop = True):
    if Tencrop:
        test_transforms = transforms.Compose([transforms.Resize(128),
                                   transforms.TenCrop(64),
                                   transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                   transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]) (crop) for crop in crops]))])

    else:
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
        
    img_data = MyDataset(img,transform = test_transforms)
    imgloader = torch.utils.data.DataLoader(img_data, batch_size=1,num_workers=0)
    return imgloader


def get_label(prediction_folder):
    labels = os.listdir(prediction_folder)
    labels = [i for i in labels if "." not in i]
    labels.sort()
    return labels


def getmodel(weight_paths):
    net = EfficientNet.from_pretrained("efficientnet-b3",num_classes=228)
    checkpoint = torch.load(weight_paths, map_location = "cpu")
    net.load_state_dict(checkpoint['net'])
    net = net.to(device)
    net.eval()
    return net

def drawresults(img,star,probability,prediction_folder):
    star_path = os.path.join(prediction_folder,star)
    star_path = glob.glob(os.path.join(star_path,"*"))[0]
    star_img = cv2.imread(star_path)
    star_img = cv2.cvtColor(star_img, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(10,10))
    columns = 2
    images = []
    images.append(img)
    images.append(star_img)

    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

    print('Prediction star: {}, its similarity is {}'.format(prediction, round(probability.item(), 2)))

def prediction(test_image_path,weight_paths,prediction_folder):
    img = detect_and_crop_face(test_image_path)
    imgloader = load_image(img)
    net = getmodel(weight_paths)
    labels = get_label(prediction_folder)
    with torch.no_grad():
        for batch_idx, inputs in enumerate(imgloader):
            inputs = inputs.to(device)
#             print(inputs.shape)
            bs, ncrops, c, h, w = np.shape(inputs) ## for ten crop
            inputs = inputs.view(-1, c, h, w)  ## for ten crop
            outputs = net(inputs)
            outputs = outputs.view(bs, ncrops, -1).mean(1)  # for ten crop
            _, predicted = outputs.max(1)
#             print(predicted)
            probabilities = torch.nn.Softmax(dim=1)(outputs)[0]
            #print(probabilities)
            index = probabilities.argmax()
            prediction = labels[index]
            probability = probabilities[index]
            
    drawresults(img,prediction,probability,prediction_folder)
    
def vectorcomparemodel(weight_paths):

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    net = EfficientNet.from_pretrained("efficientnet-b3",num_classes=228)
    checkpoint = torch.load(weight_paths, map_location = "cpu")
    net.load_state_dict(checkpoint['net'])
    net = net.eval()
    net._fc = Identity()
    net = net.to(device)
    
    return net

def vectorcompare(test_image_path,weight_paths,vector_library_path):
    net = vectorcomparemodel(weight_paths)
    vector_library = pd.read_csv(vector_library_path)
    vector_library["vectors"] = vector_library["vectors"].apply(lambda x: x.replace("[","").replace("]","").split(", "))
    vector_library["vectors"] = vector_library["vectors"].apply(lambda x: [float(i.strip()) for i in x])
    
    img = detect_and_crop_face(test_image_path)
    imgloader = load_image(img,False)
    with torch.no_grad():
        for batch_idx, inputs in enumerate(imgloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
    vector = outputs.cpu().numpy()[0]
    df = vector_library.copy()
    df["distance"] = df["vectors"].apply(lambda x: distance.euclidean(list(vector), x))
    top5 = df.sort_values(by="distance").head(5)
    w=100
    h=40
    fig=plt.figure(figsize = [20, 10] )
    for i in range(6):
        if i == 0:
            fig.add_subplot(1,6,i+1)
            plt.imshow(img)
            plt.title("test2")
        else:
            img = plt.imread(top5.iloc[i-1,0])
            fig.add_subplot(1,6,i+1)
            plt.imshow(img)
            plt.title(top5.iloc[i-1,-2])
    plt.tight_layout(True)
    plt.show()
