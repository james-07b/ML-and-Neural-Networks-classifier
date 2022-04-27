# import things
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt

# reading in data from csv....

df = pd.read_csv('data/updatedSteph.csv')
df.columns = df.columns.to_series().apply(lambda x: x.strip())


my_df = df[['tweet', 'score']]
my_df = my_df[my_df['score'] != 2]
print(my_df)
print(my_df["score"].value_counts())

sentiment_label = my_df.score.factorize()
print(sentiment_label)


tweet = my_df.tweet.values
token_machine = Tokenizer(num_words=5000)
token_machine.fit_on_texts(tweet)
encoded_docs = token_machine.texts_to_sequences(tweet)

# padding each sentence to equal length, 280 chars is max length of a tweet
padded_sequence = pad_sequences(encoded_docs, maxlen=280)

embedding_vector_length = 32

# creating Sequential model
model = Sequential()
model.add(Embedding(len(token_machine.word_index) + 1, embedding_vector_length, input_length=280))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs=5, batch_size=32)


def predict_sentiment(text):
    tw = token_machine.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=280)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])


predict_sentiment("kyrie is trash")



