from django.shortcuts import render
from django.http import HttpResponse
import json
import ast
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
      if x == '':
        return x
      x = x.replace('$', '').replace(',', '').replace(' ' , '')
      xx = x.split('-');
      if len(xx)==2:
        avg = (int(xx[0]) + float(xx[1]))/2
        return avg
      elif len(xx) ==1:
        return float(xx[0])
    return(x)




def combine_features(row):
    res = "" 
    if (row['cuisine']!=''):
        for item in row['cuisine']:
            res += ' ' + item['name'] + ' '
    res += row['description']
    return res 

def dict_key_location_id(data):
    processed_data = {}
    for item in data['data']:

        temploc = item['location_id']
        processed_data[temploc] = item
    return processed_data

def preprocess_price_rating(data):
    try:

        data['rating'] = data['rating'].fillna(2.5)
        data['price'] = data['price'].fillna('')
        data['price'] = data['price'].apply(clean_currency);

        blank_price = 0 ;
        num_of_prices=0 ;

        for ind in data.index:

            if data['price'][ind]== '':
                continue

            blank_price += data['price'][ind]
            num_of_prices +=1

        blank_price = blank_price / num_of_prices
        data["price"].replace({"": blank_price}, inplace=True)
        # print('*'*100)
        # print(data['price'])
        rate_price_df = data.set_index('location_id')
        return rate_price_df
    except Exception as e:
        print(e)


def TF_IDF_preprocessing(desc_df):
    desc_df = desc_df.set_index('location_id')

    desc_df['description'] = desc_df['description'].fillna('');
    desc_df['cuisine'] = desc_df['cuisine'].fillna('');
    desc_df['desc_cuisine']= desc_df.apply(combine_features,axis=1);
    desc_df = desc_df[['desc_cuisine']]
    tfv = TfidfVectorizer(min_df=3,
                    max_features=None,
                    strip_accents='unicode',
                    analyzer='word',
                    token_pattern=r'\w{1,}',
                    ngram_range=(1,3),
                    stop_words='english');
            #tfidfVectorizer is much more accurate than CountVectorizer().
    tfv_matrix = tfv.fit_transform(desc_df['desc_cuisine'])
    result = pd.DataFrame(tfv_matrix.toarray(),index=desc_df.index,columns=tfv.get_feature_names())
    # print(result)
    return result 