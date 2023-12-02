import pandas as pd
import os
from PIL import Image
import requests
from io import BytesIO
from numpy.random import choice
from pathlib import Path
from time import sleep
from random import uniform
import sys

SCRAPPING_DIR  = '/content/drive/MyDrive/scrapping'
DATA_DIR = '/content/drive/MyDrive/scrapping/data/'
headers={'User-Agent': 'Opera/9.80 (X11; Linux i686; Ub'
          'untu/14.10) Presto/2.12.388 Version/12.16'}

IMPT_COLS = [
        'Make', 'Model', 'Year', 'MSRP', 'Front Wheel Size (in)', 'SAE Net Horsepower @ RPM',
        'Displacement', 'Engine Type', 'Width, Max w/o mirrors (in)', 'Height, Overall (in)',
        'Length, Overall (in)', 'Gas Mileage', 'Drivetrain', 'Passenger Capacity', 'Passenger Doors',
        'Body Style']

def clean_data(dir:str):
  df = pd.read_csv(os.path.join(dir,'specs-and-pics.csv'), dtype=str, index_col=0)
  df_pic = df[df.index.str.startswith('Picture')].T
  df_spec = df.T[IMPT_COLS]

  # Cleaning Data
  df_spec['MSRP'] = df_spec['MSRP'].str.replace(',','').str.replace('$','')
  df_spec['Front Wheel Size (in)'] = df_spec['Front Wheel Size (in)'].str[:2]
  df_spec['Net Horsepower'] = df_spec['SAE Net Horsepower @ RPM'].str[:2] + '0'
  df_spec['Displacement'] = df_spec['Displacement'].str[:3].str.replace('.', '')
  df_spec.loc[df_spec['Engine Type'].str.contains('Diesel',na=False), 'Engine Type'] = 'Diesel'
  df_spec.loc[df_spec['Engine Type'].str.contains('Gas/Ethanol',na=False), 'Engine Type'] = 'Gas/Ethanol'
  df_spec.loc[df_spec['Engine Type'].str.contains('Gas/Electric',na=False), 'Engine Type'] = 'Gas/Electric'
  df_spec.loc[df_spec['Engine Type'].str.contains('Gas/E15',na=False), 'Engine Type'] = 'Gas/E15'
  df_spec.loc[df_spec['Engine Type'].str.contains('FFV',na=False), 'Engine Type'] = 'FFV'
  df_spec.loc[df_spec['Engine Type'].str.contains('Gas |Premium|Regular|Turbo',na=False), 'Engine Type'] = 'Gas'
  df_spec['Width, Max w/o mirrors (in)'] = df_spec['Width, Max w/o mirrors (in)'].str[:4]
  df_spec['Length, Overall (in)'] = df_spec['Length, Overall (in)'].str[:5]
  df_spec['City Gas Mileage (mpg)'] = df_spec['Gas Mileage'].str[:2]
  df_spec['Highway Gas Mileage (mpg)'] = df_spec['Gas Mileage'].str[12:15]
  df_spec.loc[df_spec['Drivetrain'].str.contains('Four',na=False), 'Drivetrain'] = '4'
  df_spec['Drivetrain'] = df_spec['Drivetrain'].str[0] + 'WD'
  df_spec['Body Style'] = df_spec['Body Style'].str.replace(' Car', '')
  df_spec.loc[df_spec['Body Style'].str.contains('Pickup', na=False), 'Body Style'] = 'Pickup'
  df_spec.loc[df_spec['Body Style'].str.lower().str.contains('van', na=False), 'Body Style'] = 'Van'
  df_spec['Body Style'] = df_spec['Body Style'].str.replace('Sport Utility', 'SUV')
  df_spec.drop('Gas Mileage', axis=1, inplace=True)
  df_spec.drop('SAE Net Horsepower @ RPM', axis=1, inplace=True)
  df_spec.index = df_spec.index.str.replace(r'\.\d+$', '', regex=True)
  df_pic.index = df_pic.index.str.replace(r'\.\d+$', '', regex=True)
  df_spec = df_spec[~df_spec.index.duplicated(keep='first')]
  df_pic = df_pic[~df_pic.index.duplicated(keep='first')]

  df_spec.to_csv(os.path.join(dir, 'spec-cleaned.csv'))
  df_pic.to_csv(os.path.join(dir, 'pic-url.csv'))

def random_sleep(min_time=0.4, max_time=1):
  sleep_time = uniform(min_time, max_time)
  sleep(sleep_time)

def get_pic(link:str, label:str, image_dir:str, idx:int):
  try:
    random_sleep()
    print(link)
    r = requests.get(link, timeout=10, headers=headers)
    im = Image.open(BytesIO(r.content))

    if os.path.exists(image_dir + label) == False:
      os.mkdir(image_dir + label)

    im.save(image_dir + label + f'/{label}_{idx}.jpg')
  except:
    print(f'Problem with {link}')

def save_images(df:pd.DataFrame, image_dir:str):
  for label in df.index:
    url_list = df.loc[label].dropna()
    for idx, url in enumerate(url_list, 1):
        get_pic(url, label, image_dir, idx)

if __name__ == "__main__":
  clean_data(SCRAPPING_DIR)
  df = pd.read_csv(os.path.join(SCRAPPING_DIR,'pic-url.csv'), index_col=0)
  save_images(df, DATA_DIR)