
import csv
import json
import pickle
import math
import pandas as pd
import nltk
from nltk.stem.porter import *
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, MiniBatchKMeans

def export_restaurants(rl):
  ## RESTAURANT EXPORT ##
  l = rl
  with open('restaurant_export.csv', 'w') as output_file:
      writer = csv.writer(output_file)
      writer.writerow(['name', 'street', 'city', 'city_group', 'state', 'zip', 'latitude', 'longitude',
                      'rating_level', 'avg_rating', 'price_level', 'type', 'type_2', 'nearest_metro',
                      'mline1', 'mline2', 'mline3', 'mline4', 'mline5', 'all_mline'])
      for i in range(0,len(l)):
          writer.writerow([l[i]['name'], l[i]['street'], l[i]['city'], l[i]['city_group'], l[i]['state'], l[i]['zip'], l[i]['latitude'],
            l[i]['longitude'], l[i]['rating_level'], l[i]['avg_rating'], l[i]['price_level'], l[i]['type'], l[i]['type_2'], l[i]['nearest_metro'],
            l[i]['mline1'], l[i]['mline2'], l[i]['mline3'], l[i]['mline4'], l[i]['mline5'], l[i]['all_mline']])

  ## MENU EXPORT ##
  with open('menu_export.csv', 'w') as output_file:
      writer = csv.writer(output_file)
      writer.writerow(['name', 'street', 'city', 'city_group', 'state', 'zip', 'latitude', 'longitude',
                      'rating_level', 'price_level', 'type', 'type_2', 'item', 'price', 'description'])
      for i in range(0,len(l)):
          for j in range(0,len(l[i]['menu'])):
              writer.writerow([l[i]['name'], l[i]['street'], l[i]['city'], l[i]['city_group'], l[i]['state'], l[i]['zip'],
                    l[i]['latitude'], l[i]['longitude'], l[i]['rating_level'], l[i]['price_level'], l[i]['type'], l[i]['type_2'],
                    l[i]['menu'][j]['item'], l[i]['menu'][j]['price'], l[i]['menu'][j]['description']])

def mod_r(rl):
    l = rl
    ## LOAD JSON METRO DATA ##
    with open('metros.txt') as fl: metrodf = pd.read_json(fl)
    metrodf['dist'] = ''

    ## ADD ADDITIONAL FIELDS TO RESTAURANT DICTIONARIES ##
    for i in l:
        tmp = ""
        max_pr = 0
        for w in i['menu']:
            if w['item']: tmp=tmp+','+w['item'].lower()
            if w['description']: tmp=tmp+','+w['description'].lower()
            if w['price'] and int(w['price']) > max_pr: max_pr = int(w['price'])

        # Create word list
        i['wrd_list'] = tmp

        # Set price_level
        if max_pr > 35: i['price_level'] = 'High'
        elif max_pr >= 12: i['price_level'] = 'Medium'
        else: i['price_level'] = 'Low'

        # Set rating_level
        if i['avg_rating'] == 0: i['rating_level'] = 'None'
        elif i['avg_rating'] <= 3.0: i['rating_level'] = 'Low'
        elif i['avg_rating'] < 4.0: i['rating_level'] = 'Medium'
        else: i['rating_level'] = 'High'

        if i['city'] == 'Washington':
            for x in range(0,len(metrodf.Name)):
                metrodf.set_value(x,'dist',math.sqrt(pow(metrodf.Lat[x].item() - i['latitude'],2) + pow(metrodf.Lon[x].item() - i['longitude'],2)))

            i['nearest_metro'] = metrodf.sort_values('dist', ascending=True).iloc[0]['Name']
            i['mline1'] = metrodf.sort_values('dist', ascending=True).iloc[0]['LineCode1']
            i['mline2'] = metrodf.sort_values('dist', ascending=True).iloc[0]['LineCode2']
            i['mline3'] = metrodf.sort_values('dist', ascending=True).iloc[0]['LineCode3']
            i['mline4'] = metrodf.sort_values('dist', ascending=True).iloc[0]['LineCode4']

            if metrodf.sort_values('dist', ascending=True).iloc[0]['LineCode5'] == metrodf.sort_values('dist', ascending=True).iloc[0]['LineCode5']:
                i['mline5'] = metrodf.sort_values('dist', ascending=True).iloc[0]['LineCode5']
            else:
                i['mline5'] = None

            i['all_mline'] = i['mline1']
            if i['mline2']: i['all_mline'] = '/'.join([i['all_mline'], i['mline2']])
            if i['mline3']: i['all_mline'] = '/'.join([i['all_mline'], i['mline3']])
            if i['mline4']: i['all_mline'] = '/'.join([i['all_mline'], i['mline4']])
            if i['mline5']: i['all_mline'] = '/'.join([i['all_mline'], i['mline5']])
            #print("{} nearest metro is {} on the {} line(s).".format(i['name'],i['nearest_metro'],i['all_mline']))
        else:
            i['nearest_metro'] = None
            i['mline1'] = None; i['mline2'] = None; i['mline3'] = None; i['mline4'] = None; i['mline5'] = None; i['all_mline'] = None

    return l

def print_metros_dc(rl):
    for w in rl:
        if w['city'] == 'Washington':
            print("{} nearest metro is {} on the {} line(s).".format(w['name'],w['nearest_metro'],w['all_mline']))

def export_metros():
    pass

def brad_tokenizer_test(wl):
    tokens = nltk.word_tokenize(wl.lower())
    ## NEW ##
    #stems = []
    #for item in tokens:
    #    stems.append(PorterStemmer().stem(item))
    #########
    tagged=nltk.pos_tag(tokens)
    #tagged=nltk.pos_tag(stems)
    chunkGram = r"""Chunk: {<N.+>*}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    wrds=[]
    for subtree in chunked.subtrees():
        wrds.append(" ".join([i[0] for i in subtree.leaves()]))
    #print(wrds[1:])
    return wrds[1:] # Return tokenized chunks

class ValueByKey(BaseEstimator, TransformerMixin):
    """
    Extracts a value from a dictionary by key.
    """
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, dicts):
        """Returns a list of values by key. """
        return [d[self.key] for d in dicts ]

def create_store_transforms(rl):
    l=rl
    k=15
    km = KMeans(n_clusters=k, init='k-means++')
    print("Clustering sparse data with %s" % km)

    trnsfrm = Pipeline([
        ('vbk', ValueByKey('wrd_list')),
        ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english')),
    ])
    print("Transforming text...")
    a = trnsfrm.fit_transform(l)
    with open('transforms_models/just_txt_pipeline.pickle', 'wb') as handle: pickle.dump(trnsfrm, handle)
    with open('transforms_models/just_txt_data.pickle', 'wb') as handle: pickle.dump(a, handle)
    print("Fitting model...")
    km.fit_predict(a)
    with open('transforms_models/just_txt_model.pickle', 'wb') as handle: pickle.dump(km, handle)

    trnsfrm = Pipeline([
        ('vbk', ValueByKey('wrd_list')),
        ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english', tokenizer=brad_tokenizer_test)),
    ])
    print("Transforming chunked text...")
    a = trnsfrm.fit_transform(l)
    with open('transforms_models/just_txt_chunks_pipeline.pickle', 'wb') as handle: pickle.dump(trnsfrm, handle)
    with open('transforms_models/just_txt_chunks_data.pickle', 'wb') as handle: pickle.dump(a, handle)
    print("Fitting model...")
    km.fit_predict(a)
    with open('transforms_models/just_txt_chunks_model.pickle', 'wb') as handle: pickle.dump(km, handle)

    trnsfrm = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('cuisinetype', Pipeline([
                    ('vbk', ValueByKey('type_2')),
                    ('labels', preprocessing.LabelBinarizer()),
                ])),
                # ('price_lev', Pipeline([
                #     ('vbk', ValueByKey('price_level')),
                #     ('labels2', preprocessing.LabelBinarizer()),
                # ])),
                #
                # ('rating_lev', Pipeline([
                #     ('vbk', ValueByKey('rating_level')),
                #     ('labels3', preprocessing.LabelBinarizer()),
                # ])),
                ('nlp', Pipeline([
                    ('vbk', ValueByKey('wrd_list')),
                    ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english'))
                ]))
            ]
        ))
    ])
    print("Transforming categories and text...")
    a = trnsfrm.fit_transform(l)
    with open('transforms_models/txt_cat_pipeline.pickle', 'wb') as handle: pickle.dump(trnsfrm, handle)
    with open('transforms_models/txt_cat_data.pickle', 'wb') as handle: pickle.dump(a, handle)
    print("Fitting model...")
    km.fit_predict(a)
    with open('transforms_models/txt_cat_model.pickle', 'wb') as handle: pickle.dump(km, handle)

    trnsfrm = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('cuisinetype', Pipeline([
                    ('vbk', ValueByKey('type_2')),
                    ('labels', preprocessing.LabelBinarizer()),
                ])),
                ('nlp', Pipeline([
                    ('vbk', ValueByKey('wrd_list')),
                    ('tfidf_tmp', TfidfVectorizer(lowercase=True, min_df = 2, stop_words='english', tokenizer=brad_tokenizer_test))
                ]))
            ]
        ))
    ])
    print("Transforming categories and chunked text....")
    a = trnsfrm.fit_transform(l)
    with open('transforms_models/txt_cat_chunks_pipeline.pickle', 'wb') as handle: pickle.dump(trnsfrm, handle)
    with open('transforms_models/txt_cat_chunks_data.pickle', 'wb') as handle: pickle.dump(a, handle)
    print("Fitting model...")
    km.fit_predict(a)
    with open('transforms_models/txt_cat_chunks_model.pickle', 'wb') as handle: pickle.dump(km, handle)
