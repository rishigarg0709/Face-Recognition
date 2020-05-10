from flask import Flask, request, render_template
from scipy.spatial.distance import cosine
import extractFace
import calcScore
from keras import backend as K

app = Flask(__name__)

@app.route('/')
def enter():
    return render_template('index.html')

mydict = {}

@app.route('/' , methods = ['POST'])
def submit_pic():
    if request.method == 'POST':
        f = request.files['childImage']
        path = './static/{}'.format(f.filename)
        f.save(path)

        face = extractFace.extract_face_from_image(path)
        score = calcScore.get_model_scores(face)

        ismatched = False
        imgname = ""
        imgpath = ""

        if(len(mydict) == 0):
            mydict[f.filename] = score
        else:
            for img,scr in mydict.items():
                if cosine(scr,score) <= 0.4:
                    ismatched = True
                    imgname = img
                    break

        if(ismatched):
            imgpath = './static/{}'.format(imgname)
        else:
            mydict[f.filename] = score

        result_dic = {
            'path' : path,
            'imgpath' : imgpath,
            'ismatched' : ismatched
        }

    K.clear_session()
    return render_template('index.html', result = result_dic)

if __name__ == "__main__":
    app.run(debug=True) 