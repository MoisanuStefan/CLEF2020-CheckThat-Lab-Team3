import os
import time

from flask import Flask, request, jsonify
from rq import Queue
from worker import conn
from svm import SVMAlgorithm

q = Queue(connection=conn)

app = Flask(__name__)


@app.route('/')
def init():
    print('[LOG] Synchronizing...')
    obj = SVMAlgorithm()
    obj.synchronize_predictions()
    return '<h1>Greetings from SVM Team!</h1>\n<h3>We initialized the algorithm for you so everything is ready when you need us.</h3>'


@app.route("/process/<int:count>")
def process(count):
    print('[LOG] Processing initiated for count =', count, ' tweets')
    obj2 = SVMAlgorithm()
    q.enqueue(obj2.predict, count)
    return jsonify({"response":"ok"})


if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))
    app.run(debug = True, host= '0.0.0.0', port = port)