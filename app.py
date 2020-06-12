# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:01:47 2020

@author: amreen sultana
"""

from flask import render_template, Flask, request,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import load_model
import pickle 
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('sentiment.h5')

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tpredict')
@app.route('/', methods = ['GET','POST'])

def page2():
    if request.method == 'GET':
        img_url = url_for('static',filename = 'style/3.jpg')
        return render_template('index.html',url=img_url)
    if request.method == 'POST':
        topic = request.form['tweet']
        print("Hey " +topic)
        topic=cv.transform([topic])
        print("\n"+str(topic.shape)+"\n")
        with graph.as_default():
            y_pred = cla.predict_classes(topic)
            print("pred is "+str(y_pred))
        if(y_pred[0] == 2):
            img_url = url_for('static',filename = 'style/1.jpg')
            topic = "Positive Tweet"
        elif(y_pred[0] == 0):
            img_url = url_for('static',filename = 'style/2.jpg')
            topic = "Negative Tweet"
        else:
            img_url = url_for('static',filename = 'style/3.jpg')
            print(img_url)
            topic = "Neutral Tweet"
           
        return render_template('index.html',ypred = topic)
        



if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    