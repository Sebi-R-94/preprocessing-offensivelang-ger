import emoji
import translate
import numpy as np
import re
import argparse

from deep_translator import LingueeTranslator
from somajo import SoMaJo
#from googletrans import Translator
#from translate import Translator
from emoji.unicode_codes import UNICODE_EMOJI
#s = u'\U0001f600'
#emo = UNICODE_EMOJI[s]
#trans = Translator()
#emo = emo.replace(':',"")
#emo = emo.replace('_', " ")
#translation = trans.translate(emo, dest='de')
#print(translation.text)
#print(emo)

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--input_train", type=str, help="name of the input training file" )
PARSER.add_argument("--input_test", type=str, help="name of the input test file")
PARSER.add_argument("--emojis", type=str, help="either replace, remove or leave")
PARSER.add_argument("--hashtags",type=bool, help="either True or False")
PARSER.add_argument("--casing", type=str, help="either Lowercase, Truecase or None")
PARSER.add_argument("--tokenize", type=bool, help="either True or False")
PARSER.add_argument("--output_train", type=str, help="name of the output training file")
PARSER.add_argument("--output_test", type=str, help="name of the output test file")

def read_data(path):
    labels = []
    sentences = []
    with open(path, "r", encoding=("utf8")) as train:
        for line in train:
            line = line.split()
            label = line[-2]
            labels.append(label)
            sentence = line[:-2]
            sentence = " ".join(sentence)
            sentences.append(sentence)
    return labels, sentences

def emoji_translator():
    emoji_dict = {}
    emoji_en = []
    emoji_de = []
    with open("emojis.txt", "r") as infile:
        for line in infile:
            line = line.replace("\n", "")
            emoji_en.append(line)
    with open("emojis_deutsch.txt", "r") as infile:
        for line in infile:
            line = line.replace("\n", "")
            emoji_de.append(line)
    for i in range(len(emoji_en)):
        emoji_dict[emoji_en[i]] = emoji_de[i]
    print(emoji_dict)
    return emoji_dict

def replace_emojis(text):
    #trans = LingueeTranslator(source="en", target="de")
    translations = emoji_translator()
    for i in range(len(text)):
        text[i] = text[i].replace("<", " ")
        text[i] = text[i].replace(">", " ")
        text[i] = text[i].replace("  ", " ")
        #print(text[i])
        line = text[i].split()
        for j in range(len(line)):
            if line[j].startswith('U+0'):
                emojis = line[j].split()
                for k in range(len(emojis)):
                    emojis[k] = emojis[k].replace("U", "\\U")
                    if len(emojis[k]) != 7 and emojis[k] != "Merkel":
                        emojis[k] = emojis[k].replace("+", "")
                        emojis[k] = emojis[k].encode('ASCII').decode('unicode-escape')
                        emojis[k] = UNICODE_EMOJI[emojis[k]]
                        emojis[k] = emojis[k].replace(":", "")
                        emojis[k] = emojis[k].replace("_", " ")
                        emojis[k] = translations[emojis[k]]
                        #emojis[k] = translation
                line[j] = " ".join(emojis)
                #if i == 80:
                    #print(line[j])
        text[i] = " ".join(line)
    #print(text[80])
    return(text)

def delete_emojis(text):
    for i in range(len(text)):
        text[i] = text[i].replace("<", " ")
        text[i] = text[i].replace(">", " ")
        text[i] = text[i].replace("  ", " ")
        #print(text[i])
        line = text[i].split()
        line_new = [tok for tok in line if not tok.startswith('U+0')]
        #for i in range(len(line)):
            #print("before:", tok)
            #if line[i].startswith('U+0'):
                #print("before", line)
                #line[i] = ""
                #print("after", line)
        text[i] = " ".join(line_new)
    return(text)

def prepare_unicode(text):
    for i in range(len(text)):
        text[i] = text[i].replace("<", " ")
        text[i] = text[i].replace(">", " ")
        text[i] = text[i].replace("  ", " ")
        text[i] = text[i].replace("+", "")
        
def replace_hashtags_tokenizer(text):
    tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    for i in range(len(text)):
        line = text[i].split()
        for j in range(len(line)):
            if line[j].startswith('#'):
                hashtag = []
                line[j] = line[j].replace('#', "")
                hashtag.append(line[j])
                tok_hashtag = tokenizer.tokenize_text(hashtag)
                for tok in tok_hashtag:
                    for t in tok:
                        print(t.text)
        text[i] = " ".join(line)
    return(text)
 
def replace_hashtags(text, camel_case):
    for i in range(len(text)): 
        #pattern = re.compile('([A-Z][a-z]+)')
        line = text[i].split()
        for j in range(len(line)):
            if line[j].startswith('#'):
                line[j] = line[j].replace('#', "")
                if camel_case:
                    line[j] = re.sub('([A-Z][a-z]+[A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', line[j]))
                    line[j] = line[j].lstrip()
                #print(line[j])
        text[i] = " ".join(line)
    return(text)       

def remove_usernames(text):
    for i in range(len(text)):
        line = text[i].split()
        for j in range(len(line)):
            if line[j].startswith("@"):
                #print(line[j])
                line[j] = 'User'
        text[i] = " ".join(line)
    return(text)


def tokenize(text):
    tokenizer = SoMaJo(language = "de_CMC")
    for i in range(len(text)):
        text[i] = text[i].split()
        tok = tokenizer.tokenize_text(text[i])
        tok_sent = []
        for sent in tok:
            for word in sent:
                tok_sent.append(word.text)
        text[i] = tok_sent
        
def lowercase(text):
    for i in range(len(text)):
        line = text[i].split()
        for j in range(len(line)):
            line[j] = line[j].lower()
        #print(line)
        text[i] = " ".join(line)
    return text
            
        
def run(args):
    y, sentences = read_data(args.input_train)    
    if args.emojis == 'replace':
        replace_emojis(sentences)
    elif args.emojis == 'remove':
        #print("test")
        delete_emojis(sentences)
    elif args.emojis == 'leave':
        prepare_unicode(sentences)
    if args.hashtags:
        replace_hashtags(sentences, camel_case=True)
    else:
        replace_hashtags(sentences, camel_case=False)
    remove_usernames(sentences)
    if args.casing == "lower":
        lowercase(sentences)
    if args.tokenize:
        tokenize(sentences)
    with open(args.output_train, "w", encoding="utf-8") as outfile:
        for i in range(len(sentences)):
            line = " ".join(sentences[i])
            outfile.write(line)
            outfile.write(" ")
            outfile.write(y[i])
            outfile.write("\n")
    y, sentences = read_data(args.input_test)    
    if args.emojis == 'replace':
        replace_emojis(sentences)
    elif args.emojis == 'remove':
        #print("test")
        delete_emojis(sentences)
    elif args.emojis == 'leave':
        prepare_unicode(sentences)
    if args.hashtags == False:
        replace_hashtags(sentences, True)
    else:
        replace_hashtags(sentences, False)
    remove_usernames(sentences)
    if args.casing == "lower":
        lowercase(sentences)
    if args.tokenize:
        tokenize(sentences)
    with open(args.output_test, "w", encoding="utf-8") as outfile:
        for i in range(len(sentences)):
            line = " ".join(sentences[i])
            outfile.write(line)
            outfile.write(" ")
            outfile.write(y[i])
            outfile.write("\n")
    
#y, sentences = read_data('germeval_training.txt')                       
#sentences_no_emojis = sentences
#replace_emojis(sentences)
#print(sentences[80])
#delete_emojis(sentences_no_emojis)
#print(sentences_no_emojis[80])  
#print(sentences[419])      
#replace_hashtags(sentences)
#print(sentences[1])
#remove_usernames(sentences)
#print(sentences[1])
#tokenize(sentences)
ARGS = PARSER.parse_args()
#print(ARGS)
#with open("germeval_training_processed.txt", "w", encoding="utf-8") as outfile:
#    for i in range(len(sentences)):
#        line = " ".join(sentences[i])
#        outfile.write(line)
#        outfile.write(" ")
#        outfile.write(y[i])
#        outfile.write("\n")
run(ARGS)