from django.shortcuts import render
from django.http import HttpResponse
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from . import restaurants as rest
from . import hotels
from django.views.decorators.csrf import csrf_exempt


# Create your views
# retaurn top 10 similar users
@csrf_exempt
def getSimilarRestaurants(request, location_id):

    if request.method == 'POST':
        try:


            data = json.loads(str(request.body, encoding='utf-8'))

            #print(data)
            ## mapping location-id->data
            processed_data = rest.dict_key_location_id(data)

            mouther_df = pd.DataFrame(data['data']);
            rate_price_df = rest.preprocess_price_rating(mouther_df[['location_id', 'rating', 'price']])

            # """## Add TF-IDF Features """
            tfidf_df = rest.TF_IDF_preprocessing(mouther_df[['location_id', 'description', 'cuisine']])

            cleanData = pd.merge(rate_price_df, tfidf_df, on='location_id')
            restaurants_similarities = cosine_similarity(cleanData)
            rest_similarity_df = pd.DataFrame(restaurants_similarities, index=cleanData.index, columns=cleanData.index)

            result_IDs = rest_similarity_df[location_id].sort_values(ascending=False)
            result = {}
            result['data'] = []


            for loc, sim in result_IDs[0:11].items():
                result['data'].append(processed_data[loc])

            print(result)
            return HttpResponse(json.dumps(result), 200)

        except:
            return HttpResponse('Error', 500)

@csrf_exempt
def getSimilarHotels(request, location_id):
    if request.method == 'POST':
        try:
            # file = open('hotelsresponse.txt','w')
            # file.write(str(request.body, encoding='utf-8'))
           # print(str(request.body, encoding='utf-8'))
            # print('*'*100)

            data = json.loads(str(request.body, encoding='utf-8'))
            #print(data['data'])
            processed_data = {}

            for item in data['data']:
                temploc = item['hotel']['hotelId']
                processed_data[temploc] = item;
                item['price'] = item['offers'][0]['price']['total']

            mouther_df = pd.json_normalize(data['data']);

            rate_price_df = hotels.preprocess_price_rating(mouther_df[['hotel.hotelId', 'hotel.rating', 'price']])
            # print("YESssssssssssssssssssssssssssssssssssssss")
            # """## Add TF-IDF Features """
            tfidf_df = hotels.TF_IDF_preprocessing(
                mouther_df[['hotel.hotelId', 'hotel.description.text', 'hotel.amenities']])
            cleanData = pd.merge(rate_price_df, tfidf_df, on='hotel.hotelId')
            Hotels_similarities = cosine_similarity(cleanData)
            Hotels_similarity_df = pd.DataFrame(Hotels_similarities, index=cleanData.index, columns=cleanData.index)
            # print(Hotels_similarity_df)
            result_IDs = Hotels_similarity_df[location_id].sort_values(ascending=False)
            #print(result_IDs)
            result = {}
            result['data'] = []

            for loc, sim in result_IDs[0:11].items():
                result['data'].append(processed_data[loc])

            # return HttpResponse('YES')
            #print(result);
            return HttpResponse(json.dumps(result), 200)

        except:
            return HttpResponse('Error', 500)


@csrf_exempt
def getRecommendedLocationRestaurants(request, separator):

    if request.method == 'POST':
        try:
            data = json.loads(str(request.body, encoding='utf-8'))

            #print(data)
            ## mapping location-id->data
            processed_data = rest.dict_key_location_id(data)

            mouther_df = pd.DataFrame(data['data']);
            mouther_df.drop_duplicates(subset=['location_id'], keep='first', inplace=True)
            rate_price_df = rest.preprocess_price_rating(mouther_df[['location_id', 'rating', 'price']])

            # """## Add TF-IDF Features """
            tfidf_df = rest.TF_IDF_preprocessing(mouther_df[['location_id', 'description', 'cuisine']])

            cleanData = pd.merge(rate_price_df, tfidf_df, on='location_id')
            restaurants_similarities = cosine_similarity(cleanData)
            rest_similarity_df = pd.DataFrame(restaurants_similarities, index=cleanData.index, columns=cleanData.index)
            # the change
            # result_IDs = rest_similarity_df[location_id].sort_values(ascending=False)

            sumSimilarities = {}
            iterator = 0
            for row in rest_similarity_df.itertuples():
                if iterator >= separator:
                    sumSimilarities[row[0]] = 0
                    for j in range(1, separator + 1):
                        sumSimilarities[row[0]] += row[j]
                    # print('index ',row[0],' to ', row[separator])
                iterator += 1

            marklist = sorted(sumSimilarities.items(), key=lambda x: x[1], reverse=True)
            sortdict = dict(marklist)
            result = {}
            result['data'] = []
            for loc, sim in sortdict.items():
                result['data'].append(processed_data[loc])

            print(result)
            return HttpResponse(json.dumps(result), 200)

        except:
            return HttpResponse('Error', 500)

