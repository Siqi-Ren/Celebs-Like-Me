import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.notebook import tqdm

def getstarinfo(pageurl):
    df=pd.DataFrame()
    name=pageurl.split("/")[-1].split(".")[0]
    star_url = pageurl
    star_response = requests.get(star_url)
    star_soup = BeautifulSoup(star_response.text, "html.parser")
    info=star_soup.find("span",attrs={"class":"info"}).text.strip()
    path=star_soup.findAll("a",attrs={"class":"more"})[1]["href"]
    imageurl='http://www.mingxing.com'+path
    imageurl
    image_response = requests.get(imageurl)
    image_soup = BeautifulSoup(image_response.text, "html.parser")
    imagelink=image_soup.find("div",attrs={"class":"page_photo2"}).findAll("li")
    for li in imagelink:
        link="http://www.mingxing.com"+li.findAll("a")[0]["href"]
        img_response = requests.get(link)
        img_soup = BeautifulSoup(img_response.text, "html.parser")
        imgcontainer=img_soup.findAll("div",attrs={"class": "swiper-zoom-container"})
        for i in imgcontainer[:5]:
            url=i.find("img")["src"]
            df=df.append([[name,url,info]])
    return df
    
def scrapping():
    rehgiondic={"neidi":117, "gangtai":22, "rihan":27}
    starimage=pd.DataFrame()
    for item in rehgiondic.items():
        region = item[0]
        print(region)
        endpage = item[1]
        for page in tqdm(range(endpage)):
            page=page+1
            print("scraping on page: ",page)
            pageurl=f"http://www.mingxing.com/ziliao/index?type={region}&p={page}"    
    #         print(pageurl)
            page_response = requests.get(pageurl)
            page_soup = BeautifulSoup(page_response.text, "html.parser")
            ##pageone
            starls=page_soup.find("div",attrs={"class":"page_starlist"}).findAll("li")
            star_link = ['http://www.mingxing.com'+ i.find("a")["href"] for i in starls]
            for star in tqdm(star_link):
                df=getstarinfo(star)
                starimage=starimage.append(df)
                starimage.to_csv("../data/starimage.csv", encoding="utf_8_sig")