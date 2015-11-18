import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib
from time import sleep
import re
import concurrent.futures
from multiprocessing import Pool

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def get_painting_jpg(url_pair,jpeg_dir='image_dir/saved_images/'):
    the_link,the_filename = url_pair.split(",")
    if the_link and the_filename:
        urllib.urlretrieve(the_link,jpeg_dir+the_filename+'.jpg')
        sleep(1)
        return the_link
    else:
        return "No page"

def kick_off(style):
    links_filename = style + '_links.txt'
    print "Running download for: ", style
    with open(links_filename) as f:
        painting_links_names = f.read().splitlines()
    for a_chunk in chunks(painting_links_names,len(painting_links_names)/10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(get_painting_jpg,url): url for url in a_chunk}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    #print('MY_ERROR: %r generated an exception: %s' % (url, exc))
                    pass
                else:
                    #print('MY_INFO: %r page is %d bytes' % (url, len(data)))
                    pass
    print "Done with:", style
    return links_filename

if __name__ == '__main__':

    #kick_off('impressionism')
    styles_to_scrape_parallel = [ "photorealism", "contemporary-realism", \
                                 "american-realism", "hyper-realism", "post-impressionism", \
                                 "pointillism", "cloisonnism", "fauvism", "intimism", "cubism", "cubo-futurism", \
                                 "cubo-expressionism", "tubism", "transavantgarde", "transautomatism", "mechanistic-cubism", \
                                 "futurism", "abstract-art", "abstract-expressionism","realism","impressionism"]
    my_pool = Pool(len(styles_to_scrape_parallel))
    run_results = my_pool.map( kick_off, styles_to_scrape_parallel) 
    print "COMPLETE" 
