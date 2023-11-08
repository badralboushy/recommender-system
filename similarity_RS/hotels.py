from django.shortcuts import render
from django.http import HttpResponse
import json
import ast
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def combine_features(row):
    res = ""
    if (row['hotel.amenities'] != ''):
        for item in row['hotel.amenities']:
            res += ' ' + item + ' '
    res += row['hotel.description.text']
    return res


def preprocess_price_rating(df):

    df['hotel.rating'] = df['hotel.rating'].fillna(2.5)
    df['price'] = df['price'].fillna('');
    blank_price = 0;
    num_of_prices = 0;
    for ind in df.index:
        if df['price'][ind] == '':
            continue
        price = float(df['price'][ind])
        df['price'][ind] = price
        blank_price += price
        num_of_prices += 1

    blank_price = blank_price / num_of_prices;
    df["price"].replace({"": blank_price}, inplace=True)

    rate_price_df = df.set_index('hotel.hotelId')
    return rate_price_df


def TF_IDF_preprocessing(desc_df):
    desc_df = desc_df.set_index('hotel.hotelId')

    # Removing NaN values
    desc_df['hotel.description.text'] = desc_df['hotel.description.text'].fillna('')
    desc_df['hotel.amenities'] = desc_df['hotel.amenities'].fillna('')
    desc_df['desc_amenities'] = desc_df.apply(combine_features, axis=1)
    desc_df = desc_df[['desc_amenities']]

    tfv = TfidfVectorizer(min_df=3,
                          max_features=None,
                          strip_accents='unicode',
                          analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          stop_words='english');
    # tfidfVectorizer is much more accurate than CountVectorizer().
    tfv_matrix = tfv.fit_transform(desc_df['desc_amenities'])
    result = pd.DataFrame(tfv_matrix.toarray(), index=desc_df.index, columns=tfv.get_feature_names())
    return result
