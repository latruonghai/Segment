import sys
import test
from datetime import datetime

import jyserver.Flask as jsf
from time import time
from multiprocessing import Process, Queue
import pandas as pd
from flask import current_app as app
from flask import make_response, redirect, render_template, request, url_for
from model.CVs.Segment import Segmentation
from model.Query.okapi import Okapi

from .models import Todo, db

#from preproccessing.Preprocessing import Preprocessing
#from crawl.CData import CData


file_path = ['./model/Okapi BM25/src/weight of Dataset/avgdl.pkl',
             './model/Okapi BM25/src/weight of Dataset/dl.pkl',
             './model/Okapi BM25/src/weight of Dataset/dltable.pkl',
             './model/Okapi BM25/src/weight of Dataset/file2terms.pkl',
             './model/Okapi BM25/src/weight of Dataset/files.pkl',
             './model/Okapi BM25/src/weight of Dataset/idf.pkl',
             './model/Okapi BM25/src/weight of Dataset/invertedIndex.pkl']


''' Create Server '''


@jsf.use(app)
class App():

    def __init__(self):
        self.image = ""
        print("Will create model app")
        self.q = Queue()
        self.model = "model/weight/seg_model.h5"
        # self.Seg = Segmentation(self.model)
        # self.default = self.Seg.GetModelInfo()
        #self.js.document.getElementById("btn_new").onclick =self.process()

    def process(self):
        self.image = self.js.document.getElementById("picture_1").name
        # print(self.image)
        url = f"./application/static/image/job_conver/{self.image}"
        # seg =Segmentation(url, self.model)
        # name = seg.Processing()
        start = time()
        p = Process(target=self.MiniProcess, args=(url, self.model))
        p.start()
        name = self.q.get()
        p.join()
        # seg = Segmentation(self.image, self.model)
        self.js.document.getElementById(
            "picture_2").src = f"../static/image/{name}"
        # seg.Processing()
        end = time() - start
        # self.js.prompt("Done")
        self.js.alert(f'Da xong trong {end}')

        print("Day lay use")

    def MiniProcess(self, url, model):

        # seg = Segmentation(url, model)
        seg = Segmentation(url, model)
        res = seg.Processing(url)
        self.q.put(res)


@app.route('/IR', methods=['GET', 'POST'])
def posts():
    if request.method == 'POST':
        df = pd.read_csv('Source_new.csv')
        keywords = request.form['cs']
        #keywords = request.args.get('cs')
        #keywords = 'Đà Lạt'
        que = Okapi(keywords, file_path)
        res, name = que.letQuery()
        # print(res)
        try:
            for r, n in zip(res, name):
                sources = df[df['Files name'] == n]['Sources'].tolist()[0]

                # print(sources)
                new_post = Todo(
                    ids=r['id'],
                    title=r['title'],
                    content=r['content'],
                    keyword=que.query,
                    source=sources)
                try:
                    a = Todo.query.filter_by(ids=r['id']).first()
                    # print(a)
                    a.ids = new_post.ids
                    a.title = new_post.title
                    a.content = new_post.content
                    a.keyword = new_post.keyword
                    a.source = new_post.source
                    a.date_created = new_post.date_created
                    a.completed = new_post.completed
                    db.session.commit()
                except BaseException:
                    db.session.add(new_post)
                    db.session.commit()
            return redirect('/IR')
        except (TypeError, IndexError):
            delete(Todo.query.all(), db)
            return redirect('/IR')

    else:
        all_posts = Todo.query.order_by(Todo.ids).all()
        return render_template('query.html', posts=all_posts)


@app.route('/')
def index():
    if request.method == 'GET':
        print('Hello')
    return App.render(render_template('CV.html'))
# @app.route('/I', methods=['GET', 'POST'])


@app.route('/upload_img={name}')
def indexs(name):
    if request.method == 'GET':
        request.form
        #seg = Segmentation()
        print("Done")
    return render_template('CV.html')


@app.route('/Crawl')
def crawl():
    return render_template('crawl.html', name=name)


def delete(query, db):
    for que in query:
        db.session.delete(que)
    db.session.commit()


if __name__ == "__main__":
    app.run(debug=True, use_debugger=True, use_reloader=True)
