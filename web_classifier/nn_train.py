import jieba
import numpy
import pickle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
import matplotlib.pyplot as plt
import shutil


def _dircheck(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _readfile(path):
    with open(path, 'r', encoding='utf-8') as fr:
        content = fr.read()
        return content


def _savefile(savepath, content):
    with open(savepath, "w", encoding='utf-8') as fw:
        fw.write(content)


def analyze_file(path):
    cp = 'custom' + os.sep + 'corpus'
    sp = 'custom' + os.sep + 'segment'
    if os.path.exists(cp):
        shutil.rmtree(cp)
    if os.path.exists(sp):
        shutil.rmtree(sp)
    with open(path, 'r', encoding='utf-8') as fr:
        first_line = fr.readline()
        params = json.loads(first_line)
        model = params['model']
        model_name = params['model_name']
        scale = float(params['scale'])
        line = fr.readline()
        num = 1
        while line:
            data = json.loads(line)
            content = data['content']
            classification = data['class']
            path = 'custom' + os.sep + 'corpus' + os.sep + classification
            _dircheck(path)
            file_path = path + os.sep + str(num) + '.txt'
            num += 1
            fw = open(file_path, 'w', encoding='utf-8')
            fw.write(content)
            fw.close()
            line = fr.readline()
    print('文件分类保存完成')
    return model, model_name, scale


def corpus_seg():
    corpus_path = 'custom' + os.sep + 'corpus'
    clslist = os.listdir(corpus_path)
    for cls in clslist:
        class_path = corpus_path + os.sep + cls + os.sep
        seg_path = 'custom' + os.sep + 'segment' + os.sep
        _dircheck(seg_path)
        cls_seg_path = seg_path + cls + os.sep
        _dircheck(cls_seg_path)
        one_cls_list = os.listdir(class_path)
        for single_file in one_cls_list:
            fullname = class_path + single_file
            content = _readfile(fullname)
            content = content.replace('\r\n', '')
            content = content.replace(' ', '')
            content_seg = jieba.cut(content)
            _savefile(cls_seg_path + single_file, " ".join(content_seg))  # 将处理后的文件保存到分词后语料目录
    print('分词完成！划分测试集结束！')


def creare_model(model, model_name, vec_path, val_scale):
    seg_path = 'custom' + os.sep + 'segment' + os.sep
    label_list = os.listdir(seg_path)
    label_list.sort()
    label_path = 'labels' + os.sep + model_name + '_' + model + '_' + 'labels.txt'
    with open(label_path, 'w', encoding='utf-8') as fw:
        for l in label_list:
            fw.write(l)
            fw.write(' ')
    label_num = len(label_list)  # 标签的总数，后续生成标签序列和训练需要用到
    texts = []  # 语料集的文本序列
    labels = []  # 语料集的分类id序列
    for label in label_list:
        label_id = label_list.index(label)
        path = seg_path + label
        txt_list = os.listdir(path)
        for txt in txt_list:
            txt_path = path + os.sep + txt
            texts.append(_readfile(txt_path))
            labels.append(label_id)
    labels = numpy.array(labels, dtype=int)  # 为了做后续处理和训练，必须将各个列表转换为numpy的array
    token = Tokenizer()
    token.fit_on_texts(texts)
    seq = token.texts_to_sequences(texts)  # 文本索引化序列
    word_index = token.word_index  # 词索引字典
    dic_path = 'dictionary' + os.sep
    _dircheck(dic_path)
    dic_file = dic_path + model_name + '_' + model + '_' + 'wid.pkl'
    pickle.dump(word_index, open(dic_file, 'wb'))  # 保存词索引字典
    word_num = len(word_index)  # 整个字典的词数
    print('标签抽取完毕')
    print('词索引字典生成')

    word_dic = {}  # 词向量字典
    # with open('F:/工作/2月份文本分类语料/sgns.merge.word', 'r', encoding='utf-8') as fr:
    with open(vec_path, 'r', encoding='utf-8') as fr:
        i = 0
        for line in fr:
            if i == 0:
                first_line = line.split(' ')
                word_dimension = int(first_line[1])
                i += 1
                continue
            dic_entry = line.split(' ')
            word_dic[dic_entry[0]] = numpy.asarray(dic_entry[1:word_dimension+1], dtype='float32')
            i += 1
    dic_file = dic_path + model_name + '_' + model + '_' + 'wvd.pkl'
    pickle.dump(word_dic, open(dic_file, 'wb'))  # 用户训练神经网络的时候，其实不需要保存这个字典，这条语句可删除
    print('词向量字典生成完毕')

    vec_matrix = numpy.zeros((word_num + 1, word_dimension))  # 索引-向量矩阵
    for word, index in word_index.items():
        if word in word_dic.keys():
            vec_matrix[index] = word_dic[word]
    print('矩阵生成完毕')

    labels_vec = np_utils.to_categorical(labels, label_num)
    texts_index = sequence.pad_sequences(seq, maxlen=1000, padding='post', truncating='post')
    # x_train, x_test, y_train, y_test = train_test_split(texts_index, labels_vec, test_size=0.2)  # 验证集比例
    x_train, x_test, y_train, y_test = train_test_split(texts_index, labels_vec, test_size=val_scale)  # 验证集比例
    if model == 'cnn':
        result = train_cnn(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension)
    elif model == 'lstm':
        result = train_lstm(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension)
    elif model == 'text_cnn':
        result = text_cnn(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension)
    return result


def train_lstm(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension):
    model = Sequential()
    model.add(Embedding(input_dim=dim,
                        output_dim=word_dimension,
                        mask_zero=True,
                        weights=[matrix],
                        input_length=1000,
                        trainable=False))
    model.add(Bidirectional(LSTM(units=256), merge_mode='sum'))
    model.add(Dropout(0.3))
    model.add(Dense(label_num, activation='relu'))
    print('编译模型。。。')
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('开始训练。。。')
    history = model.fit(x_train, y_train,
                        batch_size=120,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        verbose=1)
    model_file = 'neural' + os.sep + model_name + '_lstm.h5'
    model.save(model_file)
    print('评估。。。')
    loss, acc = model.evaluate(x_test, y_test, batch_size=120)
    loss_curve(history)
    result = {}
    result['loss'] = loss
    result['accuracy'] = acc
    return result


def train_cnn(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension):
    sequence_input = Input(shape=(1000,), dtype='int32')
    embedding_layer = Embedding(dim,
                                word_dimension,
                                weights=[matrix],
                                input_length=1000,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(label_num, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        nb_epoch=10, batch_size=128, verbose=1)
    model_file = 'neural' + os.sep + model_name + '_cnn.h5'
    model.save(model_file)
    loss, acc = model.evaluate(x_test, y_test, batch_size=120)
    loss_curve(history)
    result = {}
    result['loss'] = loss
    result['accuracy'] = acc
    return result


def text_cnn(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension):
    sequence_input = Input(shape=(1000,), dtype='int32')
    embedding_layer = Embedding(dim,
                                word_dimension,
                                weights=[matrix],
                                input_length=1000,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(1000 - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merged = concatenate(convs, axis=1)
    out = Dropout(0.5)(merged)
    output = Dense(32, activation='relu')(out)
    output = Dense(units=label_num, activation='sigmoid')(output)
    model = Model(sequence_input, output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        nb_epoch=10, batch_size=128, verbose=1)
    model_file = 'neural' + os.sep + model_name + '_text-cnn.h5'
    model.save(model_file)
    loss, acc = model.evaluate(x_test, y_test, batch_size=120)
    loss_curve(history)
    result = {}
    result['loss'] = loss
    result['accuracy'] = acc
    return result


def loss_curve(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def customize_train(path):
    model, model_name, scale = analyze_file(path)
    corpus_seg()
    result = creare_model(model, model_name, 'vector_file', scale)
    json_data = json.dumps(result, ensure_ascii=False)
    return json_data


def nn_prediction(content_list, model, model_name):
    if model_name.startswith('default'):
        word_index = pickle.load(open('dictionary' + os.sep + 'word_index.pkl', 'rb'))
        label_path = 'labels' + os.sep + 'default_labels.txt'
    else:
        word_index = pickle.load(open('dictionary' + os.sep + model_name + '_' + 'wid.pkl', 'rb'))
        label_path = 'labels' + os.sep + model_name + '_labels.txt'
    with open(label_path, 'r', encoding='utf-8') as fr:
        line = fr.readline()
        line = line.rstrip(' ')
        label_list = line.split(' ')
    seq_list = []
    for content in content_list:
        content = content.replace('\r\n', '')
        content = content.replace(' ', '')
        content_seg = jieba.cut(content)
        seg_string = " ".join(content_seg)
        word_list = seg_string.split(' ')
        seq = []
        for word in word_list:
            if word in word_index.keys():
                word_seq = word_index[word]
            else:
                word_seq = 0
            seq.append(word_seq)
        seq_list.append(seq)
    seq_predict = sequence.pad_sequences(seq_list, maxlen=1000, padding='post', truncating='post')
    prediction = model.predict(seq_predict)
    result = numpy.argmax(prediction, axis=1)
    id_list = result.tolist()
    result_data = []
    i = 0
    for id in id_list:
        json_dic = {}
        json_dic['content'] = content_list[i]
        json_dic['class'] = label_list[id]
        i += 1
        result_data.append(json_dic)
    json_data = json.dumps(result_data, ensure_ascii=False)
    return json_data


if __name__ == '__main__':
    content_list = []
    with open('tt.txt', 'r', encoding='utf-8') as fr:
        content_list.append(fr.readline())
    model_name = 'default_text-cnn'
    # model = load_model('neural/default_text-cnn.h5')
    result = nn_prediction(content_list, model_name)
    # result = nn_prediction(content_list, model, model_name)
    print(result)
