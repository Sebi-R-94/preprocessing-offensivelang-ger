import fasttext
import spacy
import de_core_news_sm
import argparse

import pandas as pd
import numpy as np

from spacy_sentiws import spaCySentiWS
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import time

def read_data(path):
    labels = []
    sentences = []
    with open(path, "r", encoding=("utf8")) as train:
        for line in train:
            line = line.split()
            label = line[-1]
            #print(label.lower())
            if label.lower() == "offense":
                labels.append(1)
            else:
                labels.append(0)
            sentence = line[:-1]
            sentence = " ".join(sentence)
            sentences.append(sentence)
    return labels, sentences

def get_slurs():
    slurs= []
    with open("slurs.txt", "r") as slurfile:
        for line in slurfile:
            line = line.replace("\n", "")
            slurs.append(line)
    return slurs

def get_value(sentence, words):
    sentence = sentence.split()
    for w in sentence:
        if w in words:
            return 1
    return 0

def get_binary_features(data, words):
    #print(len(data))
    binaries = np.zeros((len(data),1))
    for i in range(len(data)-1):
        binary_value = get_value(data[i], words)
        if binary_value == 1:
            binaries[i][0] = binary_value
    return binaries
         
def get_embeddings(embed_file):
    model = fasttext.load_model(embed_file)
    return model

def get_embedding_matrix(model, text):
    sentence_matrix = np.zeros((len(text), 300))
    for i in range(len(text)):
        for i, sentence in enumerate(text):
            sent_vector = model.get_sentence_vector(sentence)  
            sentence_matrix[i,] = sent_vector
    return sentence_matrix

def get_sentiment_scores(data, emoji_dict):
    nlp = de_core_news_sm.load()
    sentiws = spaCySentiWS(sentiws_path="data\sentiws")
    nlp.add_pipe(sentiws)
    scores = np.zeros((len(data), 1))
    for i in range(len(data)):
        doc = nlp(data[i])
        for j, token in enumerate(doc):
            if token._.sentiws:
                scores[i][0] += token._.sentiws
            elif str(token).startswith('U0') and len(str(token))==10:
                emoji = str(token)
                emoji = emoji.replace("U000", "0x")
                emoji = emoji.lower()
                if emoji in emoji_dict.keys():
                    scores[i][0] += emoji_dict[emoji]
    return scores

def get_emoji_dict():
    emoji_scores = pd.read_csv('Emoji_Sentiment_Data.csv')
    emoji_dict = {}
    for i, emoji in enumerate(emoji_scores['Unicode codepoint']): 
        score = -1 * (emoji_scores['Negative'][i]/emoji_scores['Occurrences'][i]) + emoji_scores['Positive'][i]/emoji_scores['Occurrences'][i]
        emoji_dict[emoji] = score
    return emoji_dict

def lemmatize(text):
    nlp = spacy.load('de_core_news_sm')
    sentences_lemmatized = []
    for sent in text:
        sentence = nlp(sent)
        sent_lemmatized = []
        for word in sentence:
            sent_lemmatized.append(word.lemma_)
        sentences_lemmatized.append(" ".join(sent_lemmatized))
    return sentences_lemmatized

        
def vectorize(datapath):
    slurs = get_slurs()
    y, sentences = read_data(datapath)
    sentences = lemmatize(sentences)
    slur_binary_features = get_binary_features(sentences, slurs)
    model = fasttext.load_model('cc.de.300.bin')
    embedding_matrix = np.zeros(shape=(len(sentences),300))
    print(embedding_matrix.shape)
    for i, sentence in enumerate(sentences):
        sent_vector = model.get_sentence_vector(sentence)
        embedding_matrix[i, ] = sent_vector
    emoji_dict = get_emoji_dict()
    sent_scores = get_sentiment_scores(sentences, emoji_dict)
    X = np.concatenate((embedding_matrix, slur_binary_features), axis = 1)
    X = np.concatenate((X, sent_scores), axis = 1)
    return X,y

def run(args):
    start_time = time.time()
    X_train,y_train = vectorize(args.train)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .1)
    #X_train = preprocessing.scale(X_train)
    #X_val = preprocessing.scale(X_val)
    param = {'C':[0.1, 1, 10, 100]}
    #param = {"C":[0.1, 1, 10, 50, 100], "gamma":['scale', 0.001, 0.01, 0.1, 1, 10]}
    svm = LinearSVC()
    svm = SVC(kernel = "linear", verbose=1)    
    clf = GridSearchCV(svm, param, scoring="f1_macro", verbose = 1)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    best_p = clf.best_params_
    print("--- %s seconds ---" % (time.time() - start_time))
    #clf = SVC(kernel='linear', C=100)
    print("--- %s seconds ---" % (time.time() - start_time))
    clf = SVC(C=best_p['C'], kernel = "linear")
    clf.fit(X_train, y_train)
    print("Finished Training!")
    print("--- %s seconds ---" % (time.time() - start_time))
    #predict = model.predict(X_val)
    #result = classification_report(y_val, predict)
    #scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
    #print(scores)
    X_test, y_test = vectorize(args.test)
    X_test = scaler.transform(X_test)
    X_test = preprocessing.scale(X_test)
    predict = clf.predict(X_test)
    #print(y_test)
    #print(predict)
    result = classification_report(y_test, predict, digits=4)
    labels_test, testfile_sents = read_data(args.test)
    name_resultfile = "result_" + args.test[22:]
    with open(name_resultfile, "w") as resultfile:
        resultfile.write("\n")
        for i in range(len(predict)):
            resultfile.write(str(testfile_sents[i]))
            resultfile.write(" ")
            if predict[i] == 1:
                resultfile.write("OFFENSE")
            elif predict[i] == 0:
                resultfile.write("OTHER")
            resultfile.write("\n")
        resultfile.write(result)
    print(result)

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--train", type=str, help="name of the train file")
PARSER.add_argument("--test", type=str, help="name of the test file")
ARGS = PARSER.parse_args()
run(ARGS)