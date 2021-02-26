import os
import sys
import json
import jieba
from sklearn.externals import joblib
from flask import Flask, request, Response
from werkzeug.utils import secure_filename
import pickle
import numpy

sys.path.append(os.path.abspath('.'))

# from web_classifier import prediction
# from web_classifier import customize_model
# from web_classifier import nn_train
import prediction
import customize_model

# import nn_train


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model_dic = {}
model_dic['default_svm'] = joblib.load('default' + os.sep + 'model' + os.sep + 'svm.m')
model_dic['default_bayes'] = joblib.load('default' + os.sep + 'model' + os.sep + 'bayes.m')
# model_dic['default_cnn'] = load_model('neural' + os.sep + 'default_cnn.h5')
# model_dic['default_lstm'] = load_model('neural' + os.sep + 'default_lstm.h5')
# model_dic['default_textcnn'] = load_model('neural' + os.sep + 'default_textcnn.h5')
cm_path = 'custom' + os.sep + 'model'
if not os.path.exists(cm_path):
    os.makedirs(cm_path)
model_list = os.listdir(cm_path)
for m in model_list:
    model_key = 'custom' + '_' + m.rstrip('.m')
    model_dic[model_key] = joblib.load(cm_path + os.sep + m)

# neural_dic = {}
# neural_path = 'neural' + os.sep
# neural_list = os.listdir(neural_path)
# for n in neural_list:
#     model_key = n.rstrip('.h5')
#     K.backend.clear_session()
#     temp_model = load_model(neural_path + n)
#     temp_model.predict(numpy.zeros((1, 1000)))
#     neural_dic[model_key] = temp_model
#     # neural_dic[model_key] = load_model(neural_path + n)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print('文件已接收，开始训练自定义模型')
        data = customize_model.custom(filepath)
        return Response(data, mimetype='application/json')


@app.route('/predict', methods=['POST'])
def predict_service():
    content_list = request.json['content']
    model = request.json['model']
    default = request.json['default']

    result_data = prediction.fast_prediction(content_list, default, model)
    json_data = json.dumps(result_data, ensure_ascii=False)
    return Response(json_data, mimetype='application/json')


@app.route('/fast_predict', methods=['POST'])
def fast_predict_service():
    content_list = request.json['content']
    model = request.json['model']
    default = request.json['default']

    seg_list = prediction.predict_segment(content_list)
    text_bunch = prediction.pack_bunch(seg_list)

    if default:
        mode = 'default'
        vec_space = prediction.vectorize(text_bunch, mode)
    else:
        mode = 'custom'
        vec_space = prediction.vectorize(text_bunch, mode, model)

    key = mode + '_' + model
    predict_model = model_dic[key]
    predict_result = predict_model.predict(vec_space.tdm)
    result_data = []
    for file_name, expct_cate in zip(vec_space.filenames, predict_result):
        json_dic = {}
        seq = int(file_name)
        json_dic['content'] = content_list[seq]
        json_dic['class'] = expct_cate
        result_data.append(json_dic)
    json_data = json.dumps(result_data, ensure_ascii=False)

    return Response(json_data, mimetype='application/json')


@app.route('/event_dist_zh', methods=['POST'])
def fast_prediction():
    """
    中文事件识别
    :return:
    """
    content_list = request.json['content']
    model = '1105_20_svm'
    default = False
    res = []
    for text in content_list:
        temp_res = customize_model.split_sentences_deal_dq(text)
        if temp_res:
            res.extend(temp_res)
    if res:
        predict_result = prediction.fast_prediction(res, default, model)
    else:
        predict_result = []
    result_data = []
    for js in predict_result:
        if js["class"] != "无":
            result_data.append(js)
    json_data = json.dumps(result_data, ensure_ascii=False)

    return Response(json_data, mimetype='application/json')


@app.route('/get_model', methods=['POST'])
def get_model():
    model_list = customize_model.get_custom_model()
    json_data = json.dumps(model_list)
    return Response(json_data, mimetype='application/json')


# @app.route('/train_nn', methods=['POST'])
# def create_nn_model():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             if not os.path.exists(UPLOAD_FOLDER):
#                 os.makedirs(UPLOAD_FOLDER)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             print('文件已接收，开始训练神经网络模型')
#         data = nn_train.customize_train(filepath)
#         return Response(data, mimetype='application/json')

# 神经网络预测，因为keras版本不对，目前没法用
# @app.route('/nn_predict', methods=['POST'])
# def nn_predict():
#     content_list = request.json['content']
#     model_name = request.json['model_name']
#     model = neural_dic[model_name]
#     json_data = nn_train.nn_prediction(content_list, model, model_name)
#     return Response(json_data, mimetype='application/json')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', threaded=False)
    app.run(host='0.0.0.0', port=4446)  # port为端口号，可自行修改
