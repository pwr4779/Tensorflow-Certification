import tensorflow_datasets as tfds

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']
trianing_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    trianing_sentences.append(str(s.numpy()))
    training_labels.append(l.npumy())
print(s.numpy())
print(l.numpy())

for s, l in train_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.npumy())