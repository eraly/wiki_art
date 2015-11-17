import pandas as pd
import urllib
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from time import sleep
import re
from multiprocessing import Pool

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:42.0) Gecko/20100101 Firefox/42.0",\
    "Referer":    "http://www.wikiart.org/en/paintings-by-style",\
    "Host":   "www.wikiart.org",\
    "Origin":   "www.wikiart.org"}

gallery_home_page='http://www.wikiart.org'

painting_descriptors = ['Title','Artist','Completion Date','Style','Period','Genre','Technique','Material','Gallery','Tags']
painting_lives_at = ['link_to','local_jpeg','jpeg_url']
painting_compo = ['from']
image_df_cols = painting_descriptors + painting_lives_at + painting_compo

def to_file_name(its_link):
    return "__".join(its_link.split("/")[-2:])

def to_soup(a_link):
    s = requests.Session()
    s.headers.update(headers)
    response = s.get(a_link)
    soup = BeautifulSoup(response.content, from_encoding='UTF-8')
    return soup

class ImageScrape(object):
    def __init__(self,gallery_type):
        self.gallery_type = gallery_type
        self.gallery_page = gallery_home_page + "/en/paintings-by-style/" + self.gallery_type
        self.paintings_df = pd.DataFrame(columns=image_df_cols)

    def build_df(self):
        for a_page in self._collect_pages():
            for a_painting_link in self._collect_links_to_paintings(a_page):
                #print a_painting_link
                a_painting_df = self._scrape_painting_link(a_painting_link)
                self.paintings_df = self.paintings_df.append(a_painting_df)
        self.paintings_df['from'] = self.gallery_type

    def save_locally(self,df_file,links_file):
        #Save pandas dataframe locally
        self.paintings_df.to_pickle(df_file)
        #Write a file with jpeg_url,jpeg_name
        with open(links_file, 'w') as f:
            for a_jpeg_url,a_jpeg_name in zip(list(self.paintings_df.jpeg_url),list(self.paintings_df.local_jpeg)):
                f.write(a_jpeg_url + ',' + a_jpeg_name + '\n')

    def _collect_pages(self):
        soup = to_soup(self.gallery_page)
        print self.gallery_page
        page_num_list = [int(x.text) for x in soup.select('.pager-items a') if x.text != 'Next' and x.text != 'Previous' and x.text != '...']
        if page_num_list:
            max_page_num = max(page_num_list)
        else:
            max_page_num = 1
        return [self.gallery_page+'/'+ str(i) for i in range(1,max_page_num+1)]

    def _collect_links_to_paintings(self,a_page):
        soup = to_soup(a_page)
        return [gallery_home_page + a_painting['href'] for a_painting in soup.select('.rimage')] 

    def _scrape_painting_link(self,a_painting_link):
        soup = to_soup(a_painting_link)
        painting_dict = {}
        painting_dict['link_to'] = [a_painting_link]
        painting_dict['jpeg_url'] = [soup.select('#paintingImage')[0]['href']]
        painting_dict['local_jpeg'] = [to_file_name(a_painting_link)]
        painting_dict['Title'] = [soup.find('h1').findChildren()[0].getText()]
        for a_row in soup.select('.DataProfileBox p'):
            k,v = "".join(a_row.getText().splitlines()).split(":")
            if k in image_df_cols:
                painting_dict[k] = [v]
        return pd.DataFrame(data=painting_dict,columns=image_df_cols)

def call_image_scrape(scrape_style):
    a_scraper = ImageScrape(a_style)
    a_scraper.build_df()
    a_scraper.build_df()
    a_scraper.save_locally(a_style+'.pkl', a_style+'_links.txt')
    

if __name__ == '__main__':

    styles_to_scrape = \
    ["photorealism", "contemporary-realism", "american-realism", "hyper-realism", "realism", "impressionism", "post-impressionism", "pointillism", "cloisonnism", "fauvism", "intimism", "cubism", "cubo-futurism", "cubo-expressionism", "tubism", "transavantgarde", "transautomatism", "mechanistic-cubism", "futurism", "abstract-art", "abstract-expressionism"]

    my_pool = Pool(40)

    run_results = my_pool.map( call_image_scrape, styles_to_scrape) 
