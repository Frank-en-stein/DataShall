##Kaggle challenge: DataShall R3 by S. M. Faisal Rahman

import csv, io, sys, re
import nltk
nltk.download('punkt')

from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class Data:
    def __init__(self, trainfile, testfile, outputfile):
        # library dependecies: csv, sys, io

        self.trainfile = trainfile
        self.training_data = [row for row in csv.reader(io.open(self.trainfile, newline='', encoding='utf-8'))]
        self.train_N = len(self.training_data)

        self.testfile = testfile
        self.test_data = [row for row in csv.reader(io.open(self.testfile, newline='', encoding='utf-8'))]
        self.test_N = len(self.test_data)

        self.outputfile = open(outputfile, 'w')
        self.predictions = []

    def clean(self):
        # library dependecies: NONE

        #stripping double quotes and setting to lower case
        self.training_data = self.training_data[1:]
        self.train_N -= 1
        for ss in self.training_data:
            ss[1] = Data.transform_sentiment(ss[1])
            Text = ss[0]
            Text = re.sub(r'\d+', ' ', Text.strip(' "\t\n\r').upper())
            Text = re.sub(r'[~!@#$%^&*()_\-+\\/,.;:"\'{}[]।]', ' ', Text)

            #handle redundant repeatation of characters
            # words = Text.split()
            # sentence = []
            # for w in words:
            #     c = Counter(w)
            #     freq = c.most_common()
            #     for x in freq:
            #         if x[1] > 2:
            #             sentence.append(w)
            #     sentence.append(w)
            # Text = ' '.join(sentence)

            if len(Text) >= 2:
                result = [Text[0], Text[1]]
                for i in range(2, len(Text)):
                    if Text[i] == Text[i - 1] and Text[i] == Text[i - 2]:
                        continue
                    result.append(Text[i])
                Text = ''.join(result)

            ss[0] = Text
            #print(ss[0])

        #same for test_data
        self.test_data = self.test_data[1:]
        self.test_N -= 1
        for ss in self.test_data:
            Text = ss[1]
            Text = re.sub(r'\d+', ' ', Text.strip(' "\t\n\r').upper())
            Text = re.sub(r'[~!@#$%^&*()_\-+\\/,.;:"\'{}[]।]', ' ', Text)

            #handle redundant repeatation of characters
            # words = Text.split()
            # sentence = []
            # for w in words:
            #     c = Counter(w)
            #     freq = c.most_common()
            #     for x in freq:
            #         if x[1]>2:
            #             sentence.append(w)
            #     sentence.append(w)
            # Text = ' '.join(sentence)

            if len(Text) >= 2:
                result = [Text[0], Text[1]]
                for i in range(2, len(Text)):
                    if Text[i] == Text[i - 1] and Text[i] == Text[i - 2]:
                        continue
                    result.append(Text[i])
                Text = ''.join(result)

            ss[1] = Text

    def write_file(self):
        self.outputfile.write('Id,Sentiment\n')
        for i in range(self.test_N):
            self.outputfile.write(self.test_data[i][0] + ',' + str(Data.transform_sentiment_back(self.predictions[i])) + '\n')

    def predict_naive_bayes(self):
        vectorizer = CountVectorizer(stop_words='english') #CountVectorizer(stop_words='english')
        train_features = vectorizer.fit_transform([r[0] for r in self.training_data])
        #print(vectorizer.vocabulary_)
        test_features = vectorizer.transform([r[1] for r in self.test_data])
        #print(train_features)
        nb = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
        nb.fit(train_features, [r[1] for r in self.training_data])

        self.predictions = nb.predict(test_features)

    def predict_random_forest(self):
        vectorizer = CountVectorizer(stop_words='english')
        train_features = vectorizer.fit_transform([r[0] for r in self.training_data])
        test_features = vectorizer.transform([r[1] for r in self.test_data])

        forest = RandomForestClassifier(n_estimators=1000, n_jobs=16, max_features='auto', min_samples_leaf=1)
        forest.fit(train_features, [r[1] for r in self.training_data])

        self.predictions = forest.predict(test_features)

    @staticmethod
    def tokenize(text):
        # library dependecies: nltk

        sentences = nltk.sent_tokenize(text)
        return sentences

    @staticmethod
    def transform_sentiment(ss):
        if float(ss)==1:
            return -1
        elif float(ss)==0:
            return 1
        else:
            return 0

    @staticmethod
    def transform_sentiment_back(ss):
        if ss == 1:
            return 0
        elif float(ss) == -1:
            return 1
        else:
            return 2



if __name__ == "__main__":
    data = Data('trainfile_r3.csv', 'testfile_r3.csv', 'output_random_forest1000.csv')
    data.clean()
    data.predict_naive_bayes()
    data.write_file()