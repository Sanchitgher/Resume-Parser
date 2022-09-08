from flask import Flask, Response, request
import json
import spacy
import re
#from NamedEntity import *
import flask
#from tika import parser
from werkzeug.utils import secure_filename
import os
import pandas as pd
import sys, fitz

def pdf_to_text(filepath):
  '''Extracts the text data from the pdf'''
  doc = fitz.open(filepath)
  text = " "
  for page in doc:
    text = text + str(page.get_text())
  return text

class NamedEntityService(object):
    model = None  # Where we keep the model when it's loaded
    
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = spacy.load('My_trainedmodel_61')
        return cls.model

    
    @classmethod
    def get_entities(cls, input):
        """For the input, get entities and return them."""
        clf = cls.get_model()
        answer_dict = dict()
        for x in clf(input).ents:
            label_name = x.label_
            text_name = re.sub('[^A-Za-z0-9]+', ' ', str(x)).strip()
            answer_dict[text_name] = label_name

        # return dict([(str(x), x.label_) for x in clf(input).ents])
        return answer_dict

def preprocess(text):
  text = "".join([s for s in text.splitlines(True) if s.strip("\r\n")])
  return text

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'
app.config['UPLOAD_EXTENSIONS'] = ['pdf']




@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route("/predict_entity", methods=["POST"])
def predict_entity():
    f = request.files['file']
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    text = pdf_to_text('uploads/'+filename)    
    entity_dict = NamedEntityService.get_entities(text)
    df = pd.DataFrame(list(entity_dict.items()),columns=['text','entity'])
    # df.to_csv('result.csv')

    return flask.render_template('result.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

    return Response(json.dumps(entity_dict), status=200, mimetype='application/json')
    


if __name__ == "__main__":
    app.run(host="0.0.0.0" ,port = 80, debug=True, threaded=True)
