import glob
import os
import pickle
from collections import Counter
from datetime import datetime

import math
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from feature_extractor import FeatureExtractor
from utils import helpers
from utils import textprocessing

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_data']
        filename = secure_filename(file.filename)
        f = filename.split('.')
        if f[1] == 'jpg':
            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
            img.save(uploaded_img_path)

            query = fe.extract(img)
            dists = np.linalg.norm(features - query, axis=1)  # Do search
            ids = np.argsort(dists)[:30]  # Top 30 results
            scores = [(dists[id], img_paths[id]) for id in ids]

            return render_template('index.html',
                                   query_path=uploaded_img_path,
                                   scores=scores)
        else:
            query = file.read().decode("utf-8")
            docs_file = os.path.join(os.getcwd(), 'data', 'docs.pickle')
            inverted_index_file = os.path.join(
                os.getcwd(), 'data', 'inverted_index.pickle')

            stopwords_file = os.path.join(os.getcwd(), 'resources', 'stopwords_en.txt')

            # Deserialize data
            with open(docs_file, 'rb') as f:
                docs = pickle.load(f)
            with open(inverted_index_file, 'rb') as f:
                inverted_index = pickle.load(f)

            stopwords = helpers.get_stopwords(stopwords_file)

            dictionary = set(inverted_index.keys())

            # Get query from command line
            # Preprocess query

            query = textprocessing.preprocess_text(query, stopwords)
            query = [word for word in query if word in dictionary]
            query = Counter(query)

            # Compute weights for words in query
            for word, value in query.items():
                query[word] = inverted_index[word]['idf'] * (1 + math.log(value))

            helpers.normalize(query)

            scores = [[i, 0] for i in range(len(docs))]
            for word, value in query.items():
                for doc in inverted_index[word]['postings_list']:
                    index, weight = doc
                    scores[index][1] += value * weight

            scores.sort(key=lambda doc: doc[1], reverse=True)

            all_docs = []
            all_scores = []
            for index, score in enumerate(scores):
                if score[1] == 0:
                    break
                all_docs.append(docs[score[0]])
                all_scores.append(score[1])
            return render_template('docindex.html',
                     query_path=secure_filename(file.filename),
                     docs=zip(all_docs,all_scores))

    else:
        return render_template('index.html')




if __name__ == "__main__":
    app.run("0.0.0.0")


