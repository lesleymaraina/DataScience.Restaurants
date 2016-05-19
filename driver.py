
from ingest.ingest import IngestSystem
import roperations
import pymongo
import os
import csv
import sqlite3
import nltk
import re
from pprint import pprint
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import math
import json
import sklearn

#nltk.download('punkt')
#nltk.download('stopwords')

"""
Use this section after making changes to the Restaurant object
"""
################################################################################
#
# cities = [{'state': 'dc', 'city': 'washington'}, {'state': 'ny', 'city': 'new-york'}, {'state': 'ca', 'city': 'san-francisco'},
#             {'state': 'pa', 'city': 'philadelphia'}, {'state': 'tx', 'city': 'austin'}, {'state': 'nc', 'city': 'charlotte'},
#             {'state': 'il', 'city': 'chicago'}, {'state': 'ga', 'city': 'atlanta'}, {'state': 'wi', 'city': 'milwaukee'}, {'state': 'ca', 'city': 'los-angeles'}]
#
#
# cities = [{'state': 'dc', 'city': 'washington'}]
# loader = IngestSystem(cities)
# loader.pull_and_load()
#
# for i in loader.final_rlist: print(i['name'])
# try:
#     with open('loaderfinalrlist.pickle', 'wb') as handle:
#         pickle.dump(loader.final_rlist, handle)
#         print("Saved final list!")
# except:
#     print("Could not save final list")
################################################################################

'''
Driver
'''

## LOAD USER INPUTS FROM FLASK ##
yn = 'n'
while yn != 'y': yn = input("Have you entered your inputs? (y/n): ")
print("Loading user inputs...")
print()
with open('user_inputs.pickle', 'rb') as handle: u = pickle.load(handle)

userlist = [{}]
print("Keywords: ",u[0]); userlist[0]['wrd_list'] = u[0]
print("Latitude: ",u[1]); userlist[0]['latitude'] = u[1]
print("Longitude: ",u[2]); userlist[0]['longitude'] = u[2]
print("Category: ",u[3]); userlist[0]['type_2'] = u[3]
print("Price Level: ",u[4]); userlist[0]['price_level'] = u[4]
print("Rating Level: ",u[5]); userlist[0]['rating_level'] = u[5]
print("Metro Line: ",u[6]); userlist[0]['metro'] = u[6]

print()
cnk = 'n'
cnk = input("Use chunking? (y/n): ")
print()

## LOAD LIST OF ALL RESTAURANT DICTIONARIES AND TRANSFORM/ADD FIELDS ##
with open('loaderfinalrlist.pickle', 'rb') as handle:
    l = roperations.mod_r(pickle.load(handle))

## DUMP MODIFIED RESTAURANT DICTIONARY LIST ##
with open('dictlist.pickle', 'wb') as handle:
    pickle.dump(l, handle)

#roperations.print_metros_dc(l) ## PRINT NEAREST RESTAURANTS IN DC AND THEIR METRO STATION ##

roperations.export_restaurants(l) ## EXPORT RESTAURANT INFO AND MENUS ##

## THIS CODE WILL CREATE RE-TRANSFORM DATA AND RE-FIT MODELS AND PICKLE RESULTS ##
yn = 'n'
yn = input("Do you need to re-transform and re-fit the models? (y/n): ")
if yn == 'y':
    print("Creating transformations and fitting models......")
    roperations.create_store_transforms(l)
    print("Done transforming and fitting models.")

## LOAD APPROPRIATE PIPELINE AND FITTED MODEL ##
print()
if userlist[0]['type_2'] == 'Suprise me!':
    if cnk == 'n':
        print("Loading transformation pipeline and model...")
        with open('transforms_models/just_txt_pipeline.pickle', 'rb') as handle: pipeln = pickle.load(handle)
        with open('transforms_models/just_txt_model.pickle', 'rb') as handle: km = pickle.load(handle)
        with open('transforms_models/just_txt_data.pickle', 'rb') as handle: datapts = pickle.load(handle)
    else:
        print("Loading transformation pipeline and model...")
        with open('transforms_models/just_txt_chunks_pipeline.pickle', 'rb') as handle: pipeln = pickle.load(handle)
        with open('transforms_models/just_txt_chunks_model.pickle', 'rb') as handle: km = pickle.load(handle)
        with open('transforms_models/just_txt_chunks_data.pickle', 'rb') as handle: datapts = pickle.load(handle)
else:
    if cnk == 'n':
        print("Loading transformation pipeline and model...")
        with open('transforms_models/txt_cat_pipeline.pickle', 'rb') as handle: pipeln = pickle.load(handle)
        with open('transforms_models/txt_cat_model.pickle', 'rb') as handle: km = pickle.load(handle)
        with open('transforms_models/txt_cat_data.pickle', 'rb') as handle: datapts = pickle.load(handle)
    else:
        print("Loading transformation pipeline and model...")
        with open('transforms_models/txt_cat_chunks_pipeline.pickle', 'rb') as handle: pipeln = pickle.load(handle)
        with open('transforms_models/txt_cat_chunks_model.pickle', 'rb') as handle: km = pickle.load(handle)
        with open('transforms_models/txt_cat_chunks_data.pickle', 'rb') as handle: datapts = pickle.load(handle)

print()
print("Done loading.")
print()

## VIEW FITTED MODEL CLUSTERS ##
labels = km.labels_ # list of label numbers
print(labels)
centroids = km.cluster_centers_ # list of cluster centers (vector)
print(centroids)

clusters = {}
n = 0
for item in labels:
    if item in clusters: clusters[item].append(l[n]['name'])
    else: clusters[item] = [l[n]['name']]
    n +=1

for item in clusters:
    print("Cluster", item)
    for i in clusters[item]: print(i)

data = {'dist_to_new': [], 'nearest_metro': [], 'name': [], 'cluster': [], 'latitude': [], 'longitude': [], 'rating_level': [], 'price_level': [], 'type_2': [], 'city_group': [], 'street': [], 'zip': [], 'all_mline': []}
n = 0
for item in labels:
    data['name'].append(l[n]['name'])
    data['rating_level'].append(l[n]['rating_level'])
    data['price_level'].append(l[n]['price_level'])
    data['type_2'].append(l[n]['type_2'])
    data['latitude'].append(l[n]['latitude'])
    data['longitude'].append(l[n]['longitude'])
    data['city_group'].append(l[n]['city_group'])
    data['street'].append(l[n]['street'])
    data['zip'].append(l[n]['zip'])
    data['all_mline'].append(l[n]['all_mline'])
    data['nearest_metro'].append(l[n]['nearest_metro'])
    data['cluster'].append(int(item))
    n +=1


## TRANSFORM NEW DATA AND PREDICT USING FITTED MODEL##
newdata = pipeln.transform(userlist)
new_lbl = km.predict(newdata)

## FIND DISTANCE BETWEEN EACH RESTAURANT DATA POINT AND THE NEW DATA POINT ##
for i in range(0,len(l)):
    ld = sklearn.metrics.pairwise.euclidean_distances(newdata, datapts[i], squared=True)
    data['dist_to_new'].append(ld[0][0].item())

print()
print("Matched cluster: ", new_lbl)
print()
print(type(new_lbl[0].item()))

df = pd.DataFrame(data)
df['dist'] = ''

## DISPLAY RESULTS ##
for i in range(0,len(df.name)):
    df.set_value(i,'dist',math.sqrt(pow(df.latitude[i].item() - userlist[0]['latitude'],2) + pow(df.longitude[i].item() - userlist[0]['longitude'],2)))

results = df[df.cluster == new_lbl[0].item()].sort_values('dist', ascending=True)  # Sort by phyiscal distance
curr_city = results['city_group'].iloc[0]
results = df[df.cluster == new_lbl[0].item()].sort_values('dist_to_new', ascending=True)  # Sort by similarity distance

results = results[results.city_group == curr_city]
print(results)

############################
# JBB - We don't want these additional filters because we don't have enough data points #
# if userlist[0]['price_level'] != 'No preference.':
#     results = results[results.price_level == userlist[0]['price_level']]
# if userlist[0]['rating_level'] != 'No preference.':
#     results = results[results.rating_level == userlist[0]['rating_level']]
# if userlist[0]['metro'] != 'No preference.':
#     results = results[results.metros.find(userlist[0]['metro'])]
# print(results)
#########################

## EXPORT FINAL RESULTS AND ALL CLUSTERS ##
writer = pd.ExcelWriter('results.xlsx'); results.to_excel(writer,'Sheet1'); writer.save()
writer = pd.ExcelWriter('all_clusters.xlsx'); df.to_excel(writer,'Sheet1'); writer.save()

'''
## MONGODC EXPORT ##
# conn=pymongo.MongoClient()
# print(conn.database_names())
# db=conn.rdata
# print(db.collection_names())
#
# os.system("mongoexport --db rdata --collection restaurants --type=csv --fields name,street,city,city_group,state,zip,latitude,longitude,type,type_2,avg_rating --out /Users/jbbinder/Desktop/new_output.csv")
#
# db.restaurants.aggregate([
#     {'$unwind': '$menu'},
#     {'$project': {'_id': 0, 'name': 1, 'street': 1, 'city': 1, 'city_group': 1, 'state': 1, 'zip': 1, 'latitude': 1, 'longitude': 1, 'cuisinetype': '$type', 'cuisinetype_2': '$type_2', 'avg_rating': 1, 'item': '$menu.item', 'price': '$menu.price', 'description': '$menu.description'}},
#     {'$out': 'aggregate_restaurants'}
# ])
#
# print(db.collection_names())
#
# os.system("mongoexport --db rdata --collection aggregate_restaurants --type=csv --fields name,street,city,city_group,state,zip,latitude,longitude,cuisinetype,cuisinetype_2,avg_rating,item,price,description --out /Users/jbbinder/Desktop/new_output_allmenuitems.csv")




## LOAD DATA INTO SQLITE AND QUERY ##
# DBPATH = '/Users/jbbinder/Desktop/sql_rdb.db'
# conn = sqlite3.connect(DBPATH)
# cur = conn.cursor()
#
# cur.execute("CREATE TABLE r (name,street,city,city_group,state,zip,latitude,longitude,cuisinetype,cuisinetype_2,avg_rating REAL,item,price INTEGER,description);")
#
# with open('/Users/jbbinder/Desktop/new_output_allmenuitems.csv','r') as fin:
#    dr = csv.DictReader(fin) # comma is default delimiter
#    to_db = [(i['name'], i['street'], i['city'], i['city_group'], i['state'], i['zip'], i['latitude'], i['longitude'], i['cuisinetype'], i['cuisinetype_2'], i['avg_rating'], i['item'], i['price'], i['description']) for i in dr]
#
# cur.executemany("INSERT INTO r (name,street,city,city_group,state,zip,latitude,longitude,cuisinetype,cuisinetype_2,avg_rating,item,price,description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
# conn.commit()
#
# cur.execute("select name, item, price from r where description like '%burger%' and price < 8")
# print(cur.fetchall())
'''
