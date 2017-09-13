import os
import string
import matplotlib.pyplot as plt
import nltk
import numpy as np
import KneserNeyBigram

def openReadText(fileName):
    oFile = open(fileName, "r")
    return oFile.read()


def openPath(path):
    dir = os.listdir(path)
    textos = list()
    for x in dir:
        textos.append(openReadText(os.path.join(path, x)))
    return textos


def openText(path):
    dir = os.listdir(path)
    pastas = list()
    for x in dir:
        if x != '.DS_Store':
            pastas.extend(openPath(os.path.join(path, x)))
    return pastas


def remPunctuation(text):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    return (text.translate(translate_table))


def get_ngrams(text, n):
    lista=list()
    for i in range(1, n + 1):
        ngrams = nltk.ngrams(text.split(), i)
        for grams in ngrams:
            print(grams)
            lista.append(grams)
    return lista

def get_trigrams(text):
    """
    lista = list()
    ngrams = nltk.ngrams(text.split(), 3, pad_left=True, pad_right=False,left_pad_symbol='<s>')
    for grams in ngrams:
        lista.append(grams)
    """
    lista = list(nltk.ngrams(text.split(), 2))

    return lista



negAndDeceptive = os.path.expanduser("~/Desktop/OPSpam/op_spam_v1.4/negative_polarity/deceptive_from_MTurk")
negAndTruth = os.path.expanduser("~/Desktop/OPSpam/op_spam_v1.4/negative_polarity/truthful_from_Web")
posAndDeceptive = os.path.expanduser("~/Desktop/OPSpam/op_spam_v1.4/positive_polarity/deceptive_from_MTurk")
posAndTruth = os.path.expanduser("~/Desktop/OPSpam/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor")

textNegAndDeceptive = openText(negAndDeceptive)
textNegAndTruth = openText(negAndTruth)
textPosAndDeceptive = openText(posAndDeceptive)
textPosAndTruth = openText(posAndTruth)



ngram = list()
for textos in textNegAndDeceptive:
    ngram.extend(get_trigrams(remPunctuation(textos.lower())))

for textos in textPosAndDeceptive:
    ngram.extend(get_trigrams(remPunctuation(textos.lower())))

for textos in textPosAndTruth:
    ngram.extend(get_trigrams(remPunctuation(textos.lower())))

for textos in textNegAndTruth:
    ngram.extend(get_trigrams(remPunctuation(textos.lower())))

freq = nltk.FreqDist(ngram)

kneser = KneserNeyBigram.KneserNeyBi(freq)
for i in kneser.samples():
    print ("{0}\t\t{1}".format(i, kneser.prob(i)))

x = freq.most_common(100)

print(x)
paraPlotar = list()

for w in x:
    paraPlotar.append((w[1]/1278, kneser.prob(w[0])))

print(kneser.prob(kneser.max()))
"""
eixoX = np.arange(100)

plt.plot(eixoX,paraPlotar)
plt.grid(True)
plt.show()
"""

tok = list()
for textos in textNegAndDeceptive:
    tok.extend(nltk.word_tokenize(remPunctuation(textos.lower())))

for textos in textPosAndDeceptive:
    tok.extend(nltk.word_tokenize(remPunctuation(textos.lower())))

for textos in textPosAndTruth:
    tok.extend(nltk.word_tokenize(remPunctuation(textos.lower())))

for textos in textNegAndTruth:
    tok.extend(nltk.word_tokenize(remPunctuation(textos.lower())))

freqUni = nltk.FreqDist(tok)

print(freqUni.max())
y = freqUni.most_common(100)

paraPlotar = list()

for w in y:
    paraPlotar.append((w[1]/1595700, kneser.probUni(w[0])))
    print(w[0])

#eixoX = np.arange(100)
eixoX = list()
for i in y:
    eixoX.append(i)

plt.plot(eixoX,paraPlotar)
plt.grid(True)
plt.show()
