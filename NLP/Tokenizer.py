import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
#소문자로 변환됨을 주의 {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'catyou': 6, 'do': 7, 'you': 8, 'think': 9, 'is': 10, 'amazing': 11}
sequences = tokenizer.texts_to_sequences(sentences)

# test_data = [
#     'i really love my dog',
#     'my dog loves my manatee'
# ]
#
# test_seq = tokenizer.texts_to_sequences(test_data)
# print(test_seq)

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences)
print(word_index)
print(sequences)
print(padded)
# padding value로 -1일 지정하여 각 시퀀스의 앞쪽으로 패딩한다
padded = pad_sequences(sequences, padding='pre', value=-1)
print(padded)
# padding value로 -1일 지정하여 각 시퀀스의 뒤쪽으로 패딩한다
padded = pad_sequences(sequences, padding='post', value=-1)
print(padded)
# 시퀀스의 최대길이를 3으로 지정하고 초과한 경우 각 시퀀스의 앞쪽에서 자른다
padded = pad_sequences(sequences, maxlen=3, truncating='pre')
print(padded)
# 시퀀스의 최대길이를 3으로 지정하고 초과한 경우 각 시퀀스의 뒷쪽에서 자른다
padded = pad_sequences(sequences, maxlen=3, truncating='post')
print(padded)

print("--------------------------------------------")
print("-------------    불용어 제거     -------------")
print("--------------------------------------------")
##불용어 제거
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import getcwd
# Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]

sentences = []
labels = []
with open(f"{getcwd()}/../Data/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)

print(len(sentences))
print(sentences[0])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
print(len(word_index))

padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)
print(label_seq)
print(label_word_index)