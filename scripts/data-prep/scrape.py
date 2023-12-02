import bs4 as bs
from urllib.request import Request, urlopen
import pandas as pd
import os
import re

SAVE_PATH = '/content/drive/MyDrive/scrapping'
headers = {'User-Agent': 'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16'}

def build_url(page, addition=''):
    return f"{page}{addition}"

def fetch(page, addition=''):
    return bs.BeautifulSoup(urlopen(build_url(page, addition),
                            headers=headers).read(),
                            'lxml')

def all_makes():
    all_list = []
    for a in fetch(website, "/new-cars").find_all("a", {"class": ""}):
        all_list.append(a['href'])

    pattern = re.compile(r'/make/new,[\w-]+')
    all_make_list = [pattern.findall(item) for item in all_list]
    all_make_list_joined = ''.join(all_list)
    a_make = pattern.findall(all_make_list_joined)
    return a_make

def make_menu(listed):
    make_menu_list = []
    for make in listed:
        for div in fetch(website, make).find_all("div", {"class": "name"}):
            make_menu_list.append(div.find_all("a")[0]['href'])
    return make_menu_list


def model_menu(listed):
    model_menu_list = []
    for make in listed:
        soup = fetch(website, make)
        for div in soup.find_all("div",{"class": "year-selector"})[0]('a'):
            model_menu_list.append(div['href'])
    model_menu_list = [i.replace('overview', 'specifications') for i in model_menu_list]
    return model_menu_list

def specs_and_photos(spec_tab):
    picture_tab = [i.replace('specifications', 'photos') for i in spec_tab]
    specifications_table = pd.DataFrame()

    for spec, pic in zip(spec_tab, picture_tab):

      try:
        soup = fetch(website, spec)
        specifications_df = pd.DataFrame(columns=[soup.find_all('div',{'id':'tcc3-global-container'})[0]('h1')[0].text[:-15]])

        specifications_df.loc['Make', :] = soup.find_all('a', {'id': 'a_bc_1'})[0].text.strip()
        specifications_df.loc['Model', :] = soup.find_all('a', {'id': 'a_bc_2'})[0].text.strip()
        specifications_df.loc['Year', :] = soup.find_all('a', {'id': 'a_bc_3'})[0].text.strip()
        specifications_df.loc['MSRP', :] = soup.find_all('span', {'class': 'msrp'})[0].text
        print(spec)

        for div in soup.find_all("div", {"class": "specs-set-item"}):
          row_name = div.find_all("span")[0].text
          row_value = div.find_all("span")[1].text
          specifications_df.loc[row_name] = row_value

      except:
        print('Error with {}.'.format(website + spec))

      try:
        fetch_pics_url = str(fetch(website, pic))

        for ix, photo in enumerate(re.findall('sml.+?_s.jpg', fetch_pics_url), 1):
            specifications_df.loc[f'Picture {ix}', :] = template + photo.replace('\\', '')

        specifications_table = pd.concat([specifications_table, specifications_df], axis=1, sort=False)

      except:
        print('Error with {}.'.format(template + photo))

    return specifications_table

def run(path):
  a = all_makes()
  b = make_menu(a)
  c = model_menu(b)
  pd.DataFrame(c).to_csv(path+'/spec-list.csv', header=None)
  d = pd.read_csv(path+'/spec-list.csv', index_col=0, header=None).values.ravel()
  e = specs_and_photos(d)
  e.to_csv(path + '/specs-and-pics.csv')

if __name__ == '__main__':
  run(SAVE_PATH)
  print('finished running scrape.py')