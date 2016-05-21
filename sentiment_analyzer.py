import pickle
import re
import os
from vectorizer import vect 

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

import numpy as np 
label = {0:'Negative', 1:'Positive'}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Hardcoded reviews. Shouldn't be too difficult to prompt user for input.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Bad review: 
example = ['I would not recommend coming here. I will never eat here again. Terrible food, slow service. Our waiter Mark was very rude']

# Average review: 
#example = ['The food was amazing, but the service was...meh. Our waiter Mark was ok']

# Great review: 
#example = ['Everything about this place was awesome--The food...amazing! The atmosphere...nice! The staff...Our waiter Mark was great!']

X = vect.transform(example)

if label[clf.predict(X)[0]] == 'Negative':
	print 'Prediction: %s\nRecommended Rating: %.1f' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*5-3)

else:
	print 'Prediction: %s\nRecommended Rating: %.1f' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*5)