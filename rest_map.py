#from dbhelper import DBHelper
from flask import Flask
from flask import render_template
from flask import request
import json
import string
import pickle

app = Flask(__name__)
#DB = DBHelper()
categories = ['Suprise me!', 'African', 'American', 'Bakery', 'Breakfast & Coffee', 'Bar Food', 'Barbecue', 'Burgers & Hot Dogs', 'Caribbean',
	'Cheesesteaks', 'Chicken', 'Chinese', 'Deli & Sandwiches', 'Desserts', 'Diner', 'European', 'Indian', 'Italian', 'Japanese', 'Korean',
	'Latin American', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Pizza', 'Seafood & Sushi', 'Southern', 'Southwestern', 'Steak',
	'Thai', 'Vegetarian', 'Vietnamese', 'Wings']

pricepref = ['No preference.', 'High', 'Medium', 'Low']

ratingpref = ['No preference.', 'High', 'Medium', 'Low']

metropref = ['No preference.', 'RD', 'BL', 'OR', 'SV', 'GR', 'YL']

@app.route('/')
def home():
	#restaurants = DB.get_all_restaurants()
	#restuarants = json.dumps(restaurants)
	#return render_template('home.html', restaurants=restaurants, categories=categories)
	return render_template('home.html', categories=categories, pricepref=pricepref, ratingpref=ratingpref, metropref=metropref)


@app.route('/add', methods=['POST'])

@app.route('/clear')

@app.route('/submit_rest', methods=['POST'])
def submit_rest():
	category = request.form.get('category')
	if category not in categories:
		return home()
	try:
		latitude = float(request.form.get('latitude'))
		longitude = float(request.form.get('longitude'))
		price = request.form.get('price')
		rating = request.form.get('rating')
		metro = request.form.get('metro')
	except ValueError:
		return home()
	#description = sanitize_string(request.form.get('description'))
	#DB.add_rest(category, latitude, longitude, description)
	description = request.form.get('description')
	print("Saving user inputs")
	with open('user_inputs.pickle', 'wb') as handle:
	    pickle.dump((request.form.get('description'), float(request.form.get('latitude')), float(request.form.get('longitude')),
			request.form.get('category'), request.form.get('price'), request.form.get('rating'), request.form.get('metro')), handle, protocol=2)
	print("User inputs saved.")
	return home()

def sanitize_string(userinput):
	whitelist = string.letters + string.digits + " !?$.,;:-'()&"
	return filter(lambda x: x in whitelist, userinput)

if __name__ == '__main__':
	app.run(port=5000, debug=True)
