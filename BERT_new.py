import torch
import argparse

import numpy as np

from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

class GermEvalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_data(path):
    labels = []
    sentences = []
    with open(path, "r", encoding=("utf8")) as train:
        for line in train:
            line = line.split()
            label = line[-1]
            if label.lower() == "offense":
                labels.append(1)
            else:
                labels.append(0)
            sentence = line[:-1]
            sentences.append(sentence)
    return labels, sentences

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def data_to_dataset(text, labels, tokenizer):
    data_encodings = tokenizer(text, truncation=True, padding=True, max_length=150,  is_split_into_words=True)   
    dataset = GermEvalDataset(data_encodings, labels)
    return dataset

def finetune_bert(train_data, val_data, model_name):
    model = BertForSequenceClassification.from_pretrained("bert-base-german-cased")
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
    
    best_accuracy = 0
    
    epochs = 2
    for epoch in range(epochs):
        model.train()
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        for step, batch in enumerate(train_loader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs[1]
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() 
        model.eval()
        losses = []
        accuracies = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs[1]
                loss = loss_fn(logits, labels)
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                losses.append(loss.item())
                acc = flat_accuracy(logits, label_ids)
                accuracies.append(acc)
        val_acc = np.mean(accuracies)
        val_loss = np.mean(losses)
        print("Validation Accuracy:")
        print(str(val_acc))
        print("\n")
        print("Validation Loss:")
        print(str(val_loss))
        print("\n")
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_name)
            best_accuracy = val_acc
    print('Finished Training!')
            
def load_bert(model_dir, test_data):
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    model = BertForSequenceClassification.from_pretrained("bert-base-german-cased")
    model.to(device)
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    losses = []
    accuracies = []
    actual_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            preds = np.argmax(logits, axis=1).flatten()
            losses.append(loss.item())
            acc = flat_accuracy(logits, label_ids)
            accuracies.append(acc)
            actual_labels.extend(label_ids)
            pred_labels.extend(preds)
    test_acc = np.mean(accuracies)
    test_loss = np.mean(losses)
    print("Test Accuracy")
    print(str(test_acc))
    print("\n")
    print("Test Loss") 
    print(str(test_loss))
    print("\n")
    results = classification_report(actual_labels, pred_labels, digits=4)
    print(results)
    return pred_labels, results
    
def run(args):
    if args.mode == "train":
        labels, sentences = read_data(args.trainfile)
        train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, labels, test_size=.1, shuffle=False)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        train_data = data_to_dataset(train_texts, train_labels, tokenizer)
        val_data = data_to_dataset(val_texts, val_labels, tokenizer)
        finetune_bert(train_data, val_data, args.model)
    elif args.mode == "eval":
        test_labels, test_sentences = read_data(args.testfile)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        tokenfile = "tokenfile_"+args.testfile[21:]
        with open(tokenfile, "w") as outfile:
            for sent in test_sentences:
                sent_str = " ".join(sent)
                sent_tok = tokenizer.tokenize(sent_str)
                sent_tok_str = " ".join(sent_tok)
                outfile.write(sent_tok_str)
                outfile.write("\n")
        test_data = data_to_dataset(test_sentences, test_labels, tokenizer)
        predict, result = load_bert(args.model, test_data)
        name_resultfile = "result_" + args.testfile[21:]
        with open(name_resultfile, "w") as resultfile:
            for i in range(len(predict)):
                sent_str = " ".join(test_sentences[i])
                resultfile.write(sent_str)
                resultfile.write(" ")
                if predict[i] == 1:
                    resultfile.write("OFFENSE")
                elif predict[i] == 0:
                    resultfile.write("OTHER")
                resultfile.write("\n")
            resultfile.write(result)
        print(result)
                
        
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--mode", type=str, help="train or eval")
PARSER.add_argument("--trainfile", type=str,help="name of trainfile")
PARSER.add_argument("--testfile", type=str, help="name of testfile")
PARSER.add_argument("--model", type=str, help="in case of testing: name of pre-trained model")

ARGS = PARSER.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
run(ARGS)