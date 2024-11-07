import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from predict import Model
from config import config
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def classify(fname):
    cfg = config['test']

    CHECKPOINT_DIR = 'checkpoint'
    # model
    NAME_MODEL = 'resnet101'
    NUM_CLASSES = cfg['NUM_CLASSES']

    DEVICE = cfg['DEVICE']

    # ckpt
    NUM_TRAIN = '0001'
    NUM_CKPT = '1'
    # data
    RESIZE = cfg['RESIZE']
    LOAD_WIDTH = cfg['LOAD_WIDTH']
    LOAD_HEIGHT = cfg['LOAD_HEIGHT']
    IMAGE = os.path.join('test', fname)

    model = Model(name_model=NAME_MODEL, nb_classes=NUM_CLASSES, load_height=LOAD_HEIGHT, load_width=LOAD_WIDTH,
                  checkpoint_dir=CHECKPOINT_DIR, num_train=NUM_TRAIN, num_ckpt=NUM_CKPT, device=DEVICE)
    label, score = model.predict(IMAGE)
    return {
        'label': label,
        'confidence': str(round(score, 2))
    }


@app.route('/predict', methods=['POST'])
@cross_origin()
def index():
    record = json.loads(request.data)
    img_name = record["img"]  # ten anh
    # print(img_name)
    result = classify(img_name)
    print(result)
    # result = {
    #     'label': 'Dog',
    #     'confidence': '99%'
    # }  # response tra ve cho frontend
    return jsonify(result)


app.run()
