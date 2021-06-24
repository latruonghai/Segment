import jyserver.Flask as jsf
from flask import current_app as app

@jsf.use(app)
class App():
    def __init__(self):
        self.image = ""
        self.model = "model/weight/seg_model.h5"
        #self.js.document.getElementById("btn_new").onclick =self.process()

    def process(self):
        self.image = self.js.document.getElementById("picture-1")

        print(self.name)
        #seg = Segmentation(self.image, self.model)
        # seg.Processing()
        print("Day lay use")


print(help(App()))
