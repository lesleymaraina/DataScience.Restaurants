
"""
An ingest module
"""

from lxml import html
import requests
import json
from bs4 import BeautifulSoup
import re

#Python 3
import urllib.request

# #Python 2
# import urllib

import pymongo
import os
import pickle

class IngestSystem(object):
    def __init__(self, cl):
        self.cities = cl

    def pull_and_load(self):
        # l = self.get_city_urls()
        # r = []
        # for city in l:
        #     print(city)
        #     one_city = self.get_restaurant_urls(city)
        #     print(one_city)
        #     # Get the 100 most popular restaurants for each city
        #     #for w in one_city[:100]:
        #     for w in one_city: ## use this version when pulling additional DC restaurants ONLY to pull all restaurants
        #         if ('menu' in w[0]) and ('kids' not in w[0]):
        #             r.append(w)
        # pickle.dump(r,open('restaurant_url_list.txt','wb'))
        # r=pickle.load(open('restaurant_url_list.txt', 'rb'))
        # print(len(r))
        # self.store_raw(r[200:300])
        self.build_database()

    def get_city_urls(self):
        url_list = []
        for i in self.cities:
            url_list.append(('http://www.allmenus.com/'+i['state']+'/'+i['city']+'/-/?sort=popular', i['city'], i['state']))
        return url_list

    def get_restaurant_urls(self, url_citystate_tuple):
        uct = url_citystate_tuple
        a = HTMLReader(uct[0])
        citysoup = a.html_to_soup()
        urllist = a.soup_to_urllist(citysoup, uct[1], uct[2])
        return urllist

    def store_raw(self, rest_list):
        for r in rest_list:
            splt = r[0].split('/')
            a = HTMLReader('http://www.allmenus.com'+r[0])
            restsoup = a.html_to_soup()
            with open("raw_data/"+splt[1]+"_"+splt[2]+"_"+splt[3]+".html", "w") as f:
                print("Writing "+splt[1]+"_"+splt[2]+"_"+splt[3]+".html")
                f.write(restsoup.prettify())

    def build_database(self):
        l = []
        for filenm in os.listdir('raw_data/'):
            if filenm != '.DS_Store':
                tmp = Restaurant(filenm).db_obj()
                if (len(tmp['menu']) >= 1) and (tmp['latitude'] != 9999) and (tmp['type'] != ""):
                    l.append(tmp)
        print(len(l))
        conn = pymongo.MongoClient()
        db = conn.rdata
        for i in l:
            print("Insert "+i['name'])
            db.restaurants.insert_one(i)
        self.final_rlist = l


class HTMLReader(object):
    def __init__(self, uct):
        self.url = uct

    def html_to_soup(self):
        html = urllib.request.urlopen(self.url).read()
        soup = BeautifulSoup(html, "lxml")
        return soup

    def soup_to_urllist(self, soup, cityname, statename):
        tmp = []
        match = '/'+statename
        for u in soup.findAll("a", href=True):
            if (u['href'])[:3] == match:
                tmp.append((u['href'], cityname, statename))
        return tmp

    def build_info(self):
        pass

    def build_menu(self):
        pass

class Restaurant(object):
    def __init__(self, filenm):
        soup = BeautifulSoup(open('raw_data/'+filenm, 'r'), "lxml")
        self.name = soup.find("h1", {"itemprop": "name"}).string.strip()
        self.street = soup.find("span", {"itemprop": "streetAddress"}).string.strip()
        self.city = soup.find("span", {"itemprop": "addressLocality"}).string.strip()
        self.state = soup.find("span", {"itemprop": "addressRegion"}).string.strip()
        self.zip = soup.find("span", {"itemprop": "postalCode"}).string.strip()

        self.lat = str(soup.find("meta", {"itemprop": "latitude"}))
        self.lng = str(soup.find("meta", {"itemprop": "longitude"}))

        self.ratings = soup.findAll(attrs = {"itemprop": "ratingValue"})

        self.msoup = soup.findAll("li")

    def db_obj(self):
        r={}
        l=[]
        r['name'] = self.name
        r['street'] = self.street
        r['city'] = self.city
        r['state'] = self.state
        r['zip'] = self.zip

        # Add geolocation information
        try:
            r['latitude'] = float(re.findall(r'"(.*?)"', self.lat)[0])
            r['longitude'] = float(re.findall(r'"(.*?)"', self.lng)[0])
        except ValueError:
            r['latitude'] = float(9999.000)
            r['longitude'] = float(9999.000)

        #Create a city group for suburb city names
        a = self.city
        if a in ['Dunwoody', 'East Point', 'Sandy Springs']:
            r['city_group'] = 'Atlanta'
        elif a in ['Alsip', 'Cicero', 'Evergreen Park', 'Harwood Heights', 'Elmwood Park']:
            r['city_group'] = 'Chicago'
        elif a in ['Hollywood', 'West Hollywood']:
            r['city_group'] = 'Los Angeles'
        elif a in ['Greenfield', 'Wauwatosa', 'West Allis']:
            r['city_group'] = 'Milwaukee'
        elif a in ['South Austin']:
            r['city_group'] = 'Austin'
        else:
            r['city_group'] = a

        # Take an average of ratings, or else assign a 2.0
        if len(self.ratings) == 0:
            r['avg_rating'] = 0.0
        else:
            num=0
            count=0
            for i in self.ratings:
                num=num+float(i['content'])
                count=count+1
            r['avg_rating'] = num/float(count)

        # Add a blank to cuisine type is missing data
        if self.msoup[0].string:
            r['type'] = self.msoup[0].string.strip()
        else:
            r['type'] = ""

        # Create a second consolidated cusine type
        if self.msoup[0].string:
            a = self.msoup[0].string.strip()
            if a in ['Ethiopian']:
                r['type_2'] = 'African'
            elif a in ['Hawaiian', 'Local/Organic', 'American (New)']:
                r['type_2'] = 'American'
            elif a in ['Breakfast', 'Bakery & Pastries', 'Coffee & Tea']:
                r['type_2'] = 'Bakery, Breakfast & Coffee'
            elif a in ['Gastropub', 'Pub Food']:
                r['type_2'] = 'Bar Food'
            elif a in ['Hot Dogs', 'Burgers']:
                r['type_2'] = 'Burgers & Hot Dogs'
            elif a in ['Dominican', 'Jamaican']:
                r['type_2'] = 'Caribbean'
            elif a in ['Asian Fusion', 'Taiwanese']:
                r['type_2'] = 'Chinese'
            elif a in ['Sandwiches', 'Deli Food']:
                r['type_2'] = 'Deli & Sandwiches'
            elif a in ['Ice Cream', 'Crepes']:
                r['type_2'] = 'Desserts'
            elif a in ['Austrian', 'British', 'Eastern European', 'Eclectic & International', 'Spanish', 'French', 'Belgian', 'Irish', 'German', 'Polish']:
                r['type_2'] = 'European'
            elif a in ['Puerto Rican', 'Brazilian', 'Central American']:
                r['type_2'] = 'Latin American'
            elif a in ['Greek']:
                r['type_2'] = 'Mediterranean'
            elif a in ['Sushi', 'Seafood']:
                r['type_2'] = 'Seafood & Sushi'
            elif a in ['Soul Food', 'Cajun & Creole']:
                r['type_2'] = 'Southern'
            elif a in ['Tex-Mex']:
                r['type_2'] = 'Southwestern'
            elif a in ['Chicago Grill']:
                r['type_2'] = 'Steak'
            elif a in ['Burmese', 'Malaysian']:
                r['type_2'] = 'Thai'
            elif a in ['Noodles']:
                r['type_2'] = 'Vietnamese'
            elif a in ['Pakistani']:
                r['type_2'] = 'Middle Eastern'
            elif a in ['Salads']:
                r['type_2'] = 'Vegetarian'
            else:
                r['type_2'] = a
        else:
            r['type_2'] = ""

        # Create menu, add blanks if either price or description fields are missing
        for i in self.msoup:
            m={}
            if i.find("span","name") or i.find("span","price") or i.find("p", "description"):
                if i.find("span","name"):
                    m["item"] = i.find("span","name").string.strip()
                else:
                    m["item"] = ""

                # For prices, set $0.00 to blanks and take the first price in a range of prices
                if i.find("span","price"):
                    tmppr = i.find("span","price").string.strip()
                    tmppr = re.sub('[$]', '', tmppr)
                    print(tmppr)
                    if '-' not in tmppr:
                        if tmppr == "" or tmppr == " ":
                            m["price"] = ""
                        elif float(tmppr) == 0:
                            m["price"] = ""
                        else:
                            m["price"] = float(tmppr)
                    else:
                        if tmppr[0:tmppr.find('-')] == "" or tmppr[0:tmppr.find('-')] == " ":
                            m["price"] = ""
                        else:
                            m["price"] = float(tmppr[0:tmppr.find('-')])
                else:
                    m["price"] = ""

                if i.find("p","description"):
                    m["description"] = i.find("p","description").string.strip()
                else:
                    m["description"] = ""
                l.append(m)
        r['menu'] = l
        return r
