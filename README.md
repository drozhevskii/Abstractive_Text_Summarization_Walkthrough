# Transformer networks for abstractive summarization
Code walkthrough of abstractive summarization with the use of RNNs or the newer networks: Transformers by [Moein Hasani](https://github.com/Moeinh77).
Original repository: https://github.com/Moeinh77/Transformer-networks-for-abstractive-summarization

Here, I try to walk through the complex code for abstractive summarization using the transformer-type neutral network.

#### Code Block #1: Importing Python Libraries
Part of the power of Python (and similar languages like R) is that there are thousands of add on packages. 
Most of these are open-source, which means that they can be imported into a Python notebook for free. 

In this code block, the code author has imported three of the most popular (and useful) packages. 
The Python package index (https://pypi.org) is the “official” location for help on Python packages, 
though more detailed sources often exist such as TensorFlow.org.
```
import re
import os
import time
import numpy as np
import pandas as pd 
import unicodedata
import tensorflow as tf
import tensorflow.keras as krs
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import csv

BUFFER_SIZE = 20000
BATCH_SIZE = 64
embedding_dim = 50 # first it was 200
```

### Reading the data and dropping useless columns

#### Code Block #2: The notebook uses the data from:
https://www.kaggle.com/shashichander009/inshorts-news-data

In order to train the model, the author uses Inshorts news short summaries dataset. It contains 55k records in the form
of news headlines and their short summaries as well as their source, time, and publication date. If we use a bigger dataset
with a similar format, we could improve the quality of summarizations even more.
```
data_unprocessed_news = pd.read_excel('data.xlsx')
data_unprocessed_news.head()
```
```
	Headline	Short	Source	Time	Publish Date
0	4 ex-bank officials booked for cheating bank o...	The CBI on Saturday booked four former officia...	The New Indian Express	09:25:00	2017-03-26
1	Supreme Court to go paperless in 6 months: CJI	Chief Justice JS Khehar has said the Supreme C...	Outlook	22:18:00	2017-03-25
2	At least 3 killed, 30 injured in blast in Sylh...	At least three people were killed, including a...	Hindustan Times	23:39:00	2017-03-25
3	Why has Reliance been barred from trading in f...	Mukesh Ambani-led Reliance Industries (RIL) wa...	Livemint	23:08:00	2017-03-25
4	Was stopped from entering my own studio at Tim...	TV news anchor Arnab Goswami has said he was t...	YouTube	23:24:00	2017-03-25
```


#### Code Block #3: Shuffling
Before doing anything further, we need to shuffle the data. Shuffling ensures that we reduce data variance and make it more
representative of all data, so that the model does not overfit in the end.

If data is not shuffled, then the split into training, test, and validation set will not be representative of all data.
```
from sklearn.utils import shuffle
# shuffling the data 
data_unprocessed_news = shuffle(data_unprocessed_news)
data_unprocessed_news.head()
```
```
Headline	Short	Source	Time	Publish Date
0	4 ex-bank officials booked for cheating bank o...	The CBI on Saturday booked four former officia...	The New Indian Express	09:25:00	2017-03-26
1	Supreme Court to go paperless in 6 months: CJI	Chief Justice JS Khehar has said the Supreme C...	Outlook	22:18:00	2017-03-25
2	At least 3 killed, 30 injured in blast in Sylh...	At least three people were killed, including a...	Hindustan Times	23:39:00	2017-03-25
3	Why has Reliance been barred from trading in f...	Mukesh Ambani-led Reliance Industries (RIL) wa...	Livemint	23:08:00	2017-03-25
4	Was stopped from entering my own studio at Tim...	TV news anchor Arnab Goswami has said he was t...	YouTube	23:24:00	2017-03-25
Headline	Short	Source	Time	Publish Date
27809	BCCI partially accepts Lodha Panel recommendat...	Following a Special General Meeting of the BCC...	PTI	23:56:00	2016-10-01
32039	Radiologists defer nation-wide indefinite strike	The Indian Radiological and Imaging Associatio...	The Financial Express	09:37:00	2016-09-03
50188	TN Dalit man killed in suspected honour-killing	A 21-year-old Dalit man was hacked to death on...	The News Minute	12:24:00	2016-03-14
10140	Diego Maradona inducted in the Italian footbal...	Former Argentine footballer Diego Maradona was...	Hindustan Times	14:21:00	2017-01-19
19548	Selfie sticks are banned from Disney theme parks	Visitors to Disney amusement parks around the ...	Reuters	18:21:00	2016-11-20
```

#### Code Block #4: We create two separate dataframes for full texts and summaries.

For some reason, the labels for the columns in the original dataset are switched. That is why, we reassign them to new 
dataframes in a flipped way. We then observe the size of dataframes to make sure they are complete.
```
summaries, longreview = pd.DataFrame(), pd.DataFrame()
summaries['short'] = data_unprocessed_news['Headline']#[:data_to_use]
longreview['long'] = data_unprocessed_news['Short']#[:data_to_use]
(summaries.shape,longreview.shape)
```
```
((55104, 1), (55104, 1))
```


### Cleaning the data for training

#### Code Block #5: Cleaning the data and replacing abbreviations

The original dataset contains a lot of abbreviations that we need to turn into separate worlds in order to feed into the sequence algorithm. Also, we need to clean all unrecognized symbols and numbers.
We create a cleaning function for that using a number of ReGex replacement statements (re.sub).
```
# replacing many abbreviations and lower casing the words
def clean_words(sentence):
    sentence = str(sentence).lower()
    sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore') # for converting é to e and other accented chars
    sentence = re.sub(r"http\S+","",sentence)
    sentence = re.sub(r"there's", "there is", sentence)
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    sentence = re.sub(r"'til", "until", sentence)
    sentence = re.sub(r"\"", "", sentence)
    sentence = re.sub(r"\'", "", sentence)
    sentence = re.sub(r' s ', "",sentence)
    sentence = re.sub(r"&39", "", sentence) # the inshorts data has this in it
    sentence = re.sub(r"&34", "", sentence) # the inshorts data has this in it
    sentence = re.sub(r"[\[\]\\0-9()\"$#%/@;:<>{}`+=~|.!?,-]", "", sentence)
    sentence = re.sub(r"&", "", sentence)
    sentence = re.sub(r"\\n", "", sentence)
    sentence = sentence.strip()
    return sentence
```

#### Code Block #6: Applying cleaning on our dataframes 

We apply the cleaning function we just created onto the two dataframes with a mapping function lambda. It takes each row of a dataframe as X and puts it into the clean_words() function as its argument. After that, it rewrites all the rows of dataframes.
```
summaries['short'] = summaries['short'].apply(lambda x: clean_words(x))
longreview['long'] = longreview['long'].apply(lambda x: clean_words(x))
longreview.head()
```
```
	                                                long
27809	following a special general meeting of the bcc...
32039	the indian radiological and imaging associatio...
50188	a yearold dalit man was hacked to death on sun...
10140	former argentine footballer diego maradona was...
19548	visitors to disney amusement parks around the ...
```

#### Code Block #7: Adding labels for start and end of each summary token

We create start and end labels to tokens in order to separate summaries if we concate them all later in one big text string. We use the lambda function again to map the labels at the start and end of each summary.
```
# adding start and end token to the senteces of label 
start_token, end_token = '<startseq>' , '<endseq>'
summaries = summaries.apply(lambda x: start_token + ' ' + x + ' ' + end_token)
summaries.head()
```
```
	                                                short
27809	<startseq> bcci partially accepts lodha panel ...
32039	<startseq> radiologists defer nationwide indef...
50188	<startseq> tn dalit man killed in suspected ho...
10140	<startseq> diego maradona inducted in the ital...
19548	<startseq> selfie sticks are banned from disne...
```

#### Code Block #8: Splitting dataframes into validation and train sets (1/10 ratio)

We take 10% of the data as the validation set and leave the rest for a train set from both full-text dataframe and summaries. We use a simple index function to filter the two datasets. Lastly, we verify that our split is correct.
```
val_split = 0.1
# train validation split
summaries_train = summaries[int(len(summaries)*val_split):]
summaries_val = summaries[:int(len(summaries)*val_split)]
longreview_train = longreview[int(len(summaries)*val_split):]
longreview_val = longreview[:int(len(summaries)*val_split)]

len(longreview_val),len(longreview_train)
```
```
(5510, 49594)
```
```
longreview_train.iloc[0], summaries_train.iloc[0]
```
```
(long    a silicon valley startup has hired a reception...
 Name: 27138, dtype: object,
 short    <startseq> silicon valley startups receptionis...
 Name: 27138, dtype: object)
```

#### Code Block #9: Finding the maximum length of questions and answers

Since the dataset has some full texts of abnormal lengths, we want to normalize them to a certain maximum, which we choose to be that 90% of sentences are shorter.

The function we create takes our dataframes for summaries and full texts as longs and shorts as well as the percentile we want to compare the lengths with. After that, it uses the comprehension lists technique to calculate the lengths of all the sentences in both dataframes. Then it takes the maximum value for the lengths of both shorts and longs and 90th percentile values. 

Lastly, we implement the function onto our training sets for summaries and full texts as shorts and longs input. We set the percentile to be 90 and we output the maximum lengths of both training sets and 90 percentile length values.

Now we choose the maximum length of summaries and full texts to be 90 percentile length values for our training process.
```
# because there are senteces with unusually long lengths, 
# we caculate the max length that 95% of sentences are shorter than that
def max_length(shorts, longs, prct):
    # Create a list of all the captions
    
    length_longs = list(len(d.split()) for d in longs)
    length_shorts = list(len(d.split()) for d in shorts)

    print('percentile {} of length of news: {}'.format(prct,np.percentile(length_longs, prct)))
    print('longest sentence: ', max(length_longs))
    print()
    print('percentile {} of length of summaries: {}'.format(prct,np.percentile(length_shorts, prct)))
    print('longest sentence: ', max(length_shorts))
    print()
    return int(np.percentile(length_longs, prct)),int(np.percentile(length_shorts, prct))

# selecting sentence length based on the percentile of data that fits in the length
max_len_news, max_len_summary= max_length(summaries_train['short'].to_list(), longreview_train['long'].to_list(), 90)


print('max-length longreview chosen for training: ', max_len_news)
print('max-length summaries chosen for training: ', max_len_summary)
```
```
percentile 90 of length of news: 60.0
longest sentence:  66

percentile 90 of length of summaries: 12.0
longest sentence:  16

max-length longreview chosen for training:  60
max-length summaries chosen for training:  12
```


### Dataset preparation

#### Code Block #10: Making a vocabulary of the words function

We create a new function to extract the corpus of all words from the summaries dataframe with a minimal occurrence (an example is 3). For that reason, we first extract all the sentences into the all_captions list. Then, we extract word by word from each sentence in the all_captions list into dictionary words and their counts (word_counts). 

Lastly, we use list comprehension to filter out all the words from the word_counts dictionary that count less than 3.
```
# making a vocabulary of the words 
def create_vocab(shorts, longs = None, minimum_repeat = 3):

    # Create a list of all the captions
    all_captions = []
    for s in shorts:
        all_captions.append(s)

    # Consider only words which occur at least minimum_occurrence times in the corpus
    word_counts = {}
    nsents = 0
    for sent in all_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= minimum_repeat]
    
    vocab = list(set(vocab))
    return vocab
```

#### Code Block #11: Extracting vocabulary from summaries trainset 

We now use the previously defined vocabulary extraction function to get the words corpus from the summaries train dataframe with minimal word occurrence of 5.

After that, we want to remove all one-character words except for "a" and "i" (since those play important connector role in a sentence). Lastly, we try to output the sorted words corpus vocabulary.
```
# each word in the vocabulary must be used in the data atleast minimum_repeat times
vocab_dec = create_vocab(summaries_train['short'].to_list(), minimum_repeat=5) # here we just use the words in vocabulary of summaries
# removing one character words from vocab except for 'a'
for v in vocab_dec:
    if len(v) == 1 and v!='a' and v!='i':
        vocab_dec.remove(v) 
        
vocab_dec = sorted(vocab_dec)[1:] # [1:] is for the '' 
vocab_dec[:10]
```
```
['<endseq>',
 '<startseq>',
 'a',
 'aa',
 'aadhaar',
 'aadhaarbased',
 'aadmi',
 'aam',
 'aamir',
 'aamirs']
```

#### Code Block #12: Extracting vocabulary from full texts trainset 

Here, we are doing the same process as in the previous block but for full texts trainset and a minimum number of occurrences of 3.
```
# each word in the vocabulary must be used in the data atleast minimum_repeat times
vocab_enc = create_vocab(longreview_train['long'].to_list(), minimum_repeat=3) # here we just use the words in vocabulary of summaries
# removing one character words from vocab except for 'a'
for v in vocab_enc:
    if len(v) == 1 and v!='a' and v!='i':
        vocab_enc.remove(v) 
        
vocab_enc = sorted(vocab_enc)[1:] # [1:] is for the '' 
vocab_enc[:10]
```
```
['****ing',
 'a',
 'a**holes',
 'aa',
 'aadat',
 'aadhaar',
 'aadhaarbased',
 'aadhaarenabled',
 'aadhaars',
 'aadhar']
```

#### Code Block #13: Tokenizing word corpora and setting lengths for encoder and decoder

"<UNK>" stands for an unknown character or symbol. We want to make sure our training words corpora do not include those. The OOV token means Out-Of-Vocabulary token and filters are just a list of all the character tokens we want to filter out.
    
We use the Keras Text Tokenizer function from the Keras library to turn our training corpora with string words into lists of tokens. In both Tokenizer functions, we indicate the filter values, which are OOV tokens, and the list of unknown characters we defined (filters). 
    
Lastly, we need to know the lengths of our resulting lists of tokens to set the encoder and decoder sizes for later steps. We add 1 at the end because the len() function starts counting from 0.
```
oov_token = '<UNK>'
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' # making sure all the last nondigit nonalphabet chars are removed
document_tokenizer = krs.preprocessing.text.Tokenizer(filters = filters,oov_token=oov_token)
summary_tokenizer = krs.preprocessing.text.Tokenizer(filters = filters,oov_token=oov_token)
document_tokenizer.fit_on_texts(vocab_enc)
summary_tokenizer.fit_on_texts(vocab_dec)#summaries_train['short'])

# caculating number of words in the vocabulary of the encoder and decoder
# They are important for positional encoding
encoder_vocab_size = len(document_tokenizer.word_index) + 1 
decoder_vocab_size = len(summary_tokenizer.word_index) + 1

# vocab_size
encoder_vocab_size, decoder_vocab_size
```
```
(32449, 8886)
```

#### Code Block #14: Make dictionaries to map index to word token for both token lists

We first create empty dictionaries for the encoder and decoder to map the index to every word token. Then, we make sure that 0 indexes in our empty dictionaries are taken by padding values.

After that, we run 2 loops to populate two dictionaries with word tokens (starting to fill with index 1) and assign the associated word token to every index.
```
ixtoword_enc = {} # index to word dic
ixtoword_dec = {} # index to word dic

wordtoix_enc = document_tokenizer.word_index # word to index dic
ixtoword_enc[0] = '<PAD0>' # no word in vocab has index 0. but padding is indicated with 0
ixtoword_dec[0] = '<PAD0>' # no word in vocab has index 0. but padding is indicated with 0

for w in document_tokenizer.word_index:
    ixtoword_enc[document_tokenizer.word_index[w]] = w
################################################
wordtoix_dec = summary_tokenizer.word_index # word to index dic

for w in summary_tokenizer.word_index:
    ixtoword_dec[summary_tokenizer.word_index[w]] = w
```

#### Code Block #15: Create input and target sequences from word tokens

We now create sequences from word tokens with texts_to_sequences() function from Keras. We do it for both training and validation sets of input and target sequences to later pass into our model.
```
inputs = document_tokenizer.texts_to_sequences(longreview_train['long'])
targets = summary_tokenizer.texts_to_sequences(summaries_train['short'])
inputs_val = document_tokenizer.texts_to_sequences(longreview_val['long'])
targets_val = summary_tokenizer.texts_to_sequences(summaries_val['short'])
```

#### Code Block #16: Add padding and truncate sequences exceeding the maximum length

Now, we want to set the maximum length for input and target sequences to values defined earlier. Also, we want to add padding to sequences to ensure that they are all same lengths (post means we add padding at the end of the sequence, same for truncation). 
```
inputs = krs.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len_news, padding='post', truncating='post')
targets = krs.preprocessing.sequence.pad_sequences(targets, maxlen=max_len_summary, padding='post', truncating='post')
inputs_val = krs.preprocessing.sequence.pad_sequences(inputs_val, maxlen=max_len_news, padding='post', truncating='post')
targets_val = krs.preprocessing.sequence.pad_sequences(targets_val, maxlen=max_len_summary, padding='post', truncating='post')
```

#### Code Block #17: Shuffle the training and validation data

We use shuffle() function on input and target sequences as a sliding window of buffer size (20000). After that, we batch our data into groups with sizes of 64. Since validation sequences are much smaller than training ones, we use increased size for our batches after shuffling the data.
```
# validate train split
dataset = tf.data.Dataset.from_tensor_slices((inputs,targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_val = tf.data.Dataset.from_tensor_slices((inputs_val,targets_val)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE*2)
```
```
longreview_val.reset_index(inplace=True, drop=True)
summaries_val.reset_index(inplace=True, drop=True)
longreview_train.reset_index(inplace=True, drop=True)
summaries_train.reset_index(inplace=True, drop=True)
```


### Defining the model

#### Code Block #18: Create a function to graph validation loss

We use matplotlib library to create hist() function to graph and to keep track of validation loss of our results later.
```
import matplotlib.pyplot as plt

def hist(history):
    plt.title('Loss')

    x= [i[0] for i in history['val']]
    y=[i[1] for i in history['val']]
    plt.plot(x,y,'x-')
    
    x= [i[0] for i in history['train']]
    y=[i[1] for i in history['train']]    
    plt.plot(x,y,'o-')

    plt.legend(['validation','train'])
    plt.show()
    print('smallest val loss:', sorted(history['val'],key=lambda x: x[1])[0])
```


#### Scaled Dot Product

#### Code Block #19: Scaled dot product attention function

"An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key."[1]

Queries, keys, and values matrices are all derived from every single word token vector in a sentence. These matrices are used in a series of linear calculations of dot scaled product to get attention vectors of the same size as input word token vectors. These new attention vectors represent embedded contextual information of every word token.

"In practice, we compute the attention function on a set of queries simultaneously, packed together
into a matrix Q. The keys and values are also packed together into matrices K and V."[1]

There are generally two types of computing attention. They are Additive Attention and Scaled Dot Product Attention. The main difference between two is that the more dimensions (of features) we work with, the less performance the Additive method shows. The Scaled Dot Product uses square root of dimensions number in denominator, which makes it scale for larger dimension values. That is why we will implement that method.

https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

Below, you can see how we implement the above-mentioned equation.
```
# the job of this function is to calculate the above equation
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights
```


#### Multi-headed attention

#### Code Block #20: Class for Multi Head Attention Block

Multi Headed attention is a block in the Transformer algorithm we are using that simply stacks attention layers for each work token vector. Instead of calculating one attention representation of a word, we do multiple for each word to account for every relationship in a sentence.

We implement this whole block as a class with a set of functions that do calculations. The number of "heads" is derived from the number of dimensions for each word token vector (as it represents the number of relationships of that word with others in a sentence). 

We input all 3 layers mentioned in the previous block of code and create a single dense layer to concate attention on them later.
We create split_heads() function to split the matrices according to our batch size to feed to the algorithm in groups and not as a single input. We then use scaled_dot_product_attention() in a call() function to calculate weights of word tokens matrices in relationship to every word in a sentence, which are key matrices. 

Lastly, we multiply these weight matrices by the original values input matrices of word token vectors to get attention output vectors. However, we need to keep the dimension of the output vector the same as the input, so we run it through a dense linear layer to concate all into a single attention layer for a word token vector.
```
class MultiHeadAttention(krs.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads # The dimensions of Q, K, V are called depth

        # the input of these 3 layers are the same: X
        self.wq = krs.layers.Dense(d_model,kernel_regularizer=krs.regularizers.l2(l=lmbda_l2))
        self.wk = krs.layers.Dense(d_model,kernel_regularizer=krs.regularizers.l2(l=lmbda_l2))
        self.wv = krs.layers.Dense(d_model,kernel_regularizer=krs.regularizers.l2(l=lmbda_l2))

        self.dense = tf.keras.layers.Dense(d_model,kernel_regularizer=krs.regularizers.l2(l=lmbda_l2))
    
    # reshape the Q,K,V 
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        # learn the Q,K,V matrices (the layers' weights)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # reshape them
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # the last dens layer expect one vector so we use concat
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
            
        return output, attention_weights
```

#### Code Block #21: Positional Encoding functions

Positional encoding is used to enhance word token embedding vectors. If we do not save the information on the position of words in a sentence, we can run into contextual issues.

The function we define in the code below takes d_model which represents the number of dimensions of the input vector, and the word token index to calculate the position embeddings.

#### Positional encoding
```
def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

# The dimension of positional encodings is the same as
# the embeddings (d_model) for facilitating the summation of both.
def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

```


### Layers

#### Embeddings preparation

#### Code Block #22: Embedding layer definition

First of all, to create embeddings of our words we need to install a GloVe (Global Vectors for Word Representation). It is a popular and very powerful algorithm for word vector learning. The algorithm takes a corpus of N number of words as input and creates a co-occurrence matrix that is N x N. The training objective of GloVe is to learn word vectors such that their dot product equals the logarithm of the word’s probability of co-occurrence. 

In the function, we load and use a file that contains a set of pre-trained word embeddings with 50 dimensions (as indicated in the filename). We create a word embedding matrix of the same size as the input vocabulary to the function. Since we do not have a huge vocabulary, for any word that does not have pre-trained embeddings from the file, we use 0 padding.

Lastly, we use this new function to create word embeddings for the encoder and decoder given their vocabularies and word tokens with indexes.
```
 # Making the embedding mtrix
def make_embedding_layer(vocab_len, wordtoix, embedding_dim=200, glove=True, glove_path= '../glove'):
    if glove == False:
        print('Just a zero matrix loaded')
        embedding_matrix = np.zeros((vocab_len, embedding_dim)) # just a zero matrix 
    else:
        print('Loading glove...')
        #glove_dir = glove_path
        notebook_path = os.path.abspath('main-github.ipynb')
        embeddings_index = {} 
        #f = open(os.path.join(notebook_path, 'glove.6B.50d.txt'), encoding="utf-8")
        f = open(os.path.join(os.path.dirname(notebook_path), 'glove/glove.6B.50d.txt'), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        # Get n-dim dense vector for each of the vocab_rocc
        embedding_matrix = np.zeros((vocab_len, embedding_dim)) # to import as weights for Keras Embedding layer
        for word, i in wordtoix.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
        
        print("GloVe ",embedding_dim, ' loaded!')

    embedding_layer = Embedding(vocab_len, embedding_dim, mask_zero=True, trainable=False) # we have a limited vocab so we 
                                                                                           # do not train the embedding layer
                                                                                           # we use 0 as padding so => mask_zero=True
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])
    return embedding_layer

embeddings_encoder = make_embedding_layer(encoder_vocab_size, wordtoix_enc, embedding_dim=embedding_dim, glove=True)
embeddings_decoder = make_embedding_layer(decoder_vocab_size, wordtoix_dec, embedding_dim=embedding_dim, glove=True)
```
```
Loading glove...
GloVe  50  loaded!
Loading glove...
GloVe  50  loaded!
```


### Transformer layers

#### Code Block #23: Define hyperparameters 

We define initial learning rate (init_lr) to be 1e-3, which controls the weights at the end of each batch. We define L2 regularization parameter (lmbda_l2) to be 0.1, which affects the generalization of the data.
We define the dropout rate (d_out_rate) to be 0.1, which also helps avoid overfitting and increase validation accuracy.
We define the number of layers of our model (num_layers) to be 4.
The number of dimensions of our data is already defined to be embeddings_dim
We define the number of neurons in the feed-forward network (used for linearization of the multi-head attention layer's output).
We define the number of heads in multi-head attention layer to be 5.
```
# hyper-params
init_lr = 1e-3
lmbda_l2 = 0.1
d_out_rate = 0.1 # tested 0.4, 0.3, 0.1 values this 0.1 seems to be the best
num_layers = 4 # chaged from 4 to 5 to learn better
d_model = embedding_dim # d_model is the representation dimension or embedding dimension of a word (usually in the range 128–512)
dff = 512 # number of neurons in feed forward network
num_heads = 5 # first it was 8 i chenged it to 10 to use embd =300d
```

#### Code Block #24: Define point-wise feed-forward network block

This is the block we use at the end of both encoder and decoder blocks to normalize and make the attention output into a linear one.
We create two dense layers using Sequential architecture: one with a size defined in hyperparameters and another one with the same size as the input. We use ReLU as the activation function and the L2 regularization parameter that we defined.
```
# The Point-wise feed-forward network block is essentially a 
# two-layer linear transformation which is used identically throughout the model
def point_wise_feed_forward_network(d_model, dff):
    return krs.Sequential([
        krs.layers.Dense(dff, activation='relu',kernel_regularizer=krs.regularizers.l2(l=lmbda_l2)),
        krs.layers.Dense(d_model,kernel_regularizer=krs.regularizers.l2(l=lmbda_l2))
    ])
```

#### Code Block #25: Define a class for Encoder layer

This layer (as seen from the encoder architecture in the picture above) consists of embeddings, one multi-head attention layer, a normalization and dropout layer, and a feed-forward network layer.

In a call method, we first get the attention output from the multi-head attention layer (mha) we then use the first dropout layer. and save it as output 1. After that, we run this output through the feed-forward network and use dropout again. This way, we make sure that we normalize the output of the block and make it easier to use for future weights training.
```
class EncoderLayer(krs.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=d_out_rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = krs.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = krs.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = krs.layers.Dropout(rate)
        self.dropout2 = krs.layers.Dropout(rate)
   
    # it has 1 layer of multi-headed attention
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
```

#### Code Block #26: Define a class for the Decoder layer

Again, as seen from the architecture of a Transformer, we define this layer with two layers of multi-head attention, a few normalization and dropout layers, and a single feed-forward network. 

In a call function, we feed the first multi-head attention layer with data and run it through the first layers of dropout and normalization to get the first output (out1). After that, we use our embedding output from the encoder to feed back into the second multi-head attention layer, normalize it, and concate with output 1 afterward to get the second output (out2).

Finally, we want to run the second output through the feed-forward network and then concate it back with the second output. We do that to avoid vanishing gradients later while backpropagating.
```
class DecoderLayer(krs.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=d_out_rate):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = krs.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = krs.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = krs.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = krs.layers.Dropout(rate)
        self.dropout2 = krs.layers.Dropout(rate)
        self.dropout3 = krs.layers.Dropout(rate)
    
    # it has 2 layers of multi-headed attention
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2
```

#### Code Block #27: Define a class for Encoder block 

We create a class for a stacked encoder block that uses multiple encoder layers (num_layers) in the process. For the data input, we define and fill word embeddings and positional encoding values (pos_encoding). We also define all the encoder layers as well as dropout layers.

In the call function, we define the length of an input sequence (as in batch size) and input embedding data as x. We then run our word embeddings through the dropout layer and combine them with their positional encodings. 

Lastly, after running through another dropout layer, we run the embedding data through all the encoder layers in a for-loop to get the final concated output.
```
class Encoder(krs.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=d_out_rate):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = embeddings_encoder
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = krs.layers.Dropout(rate)
        self.dropout_embd = krs.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x = self.dropout_embd(x, training=training) # dropout added to encoder input changed from nothing to this
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
    
        return x
```

#### Code Block #28: Define a class for the Decoder block 

The same logic and structure are applied to the decoder block. 

Instead of the initial data, we feed the decoder with the encoder output and run it through all the decoder layers to get the attention weights for the final output.
```
class Decoder(krs.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=d_out_rate):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = embeddings_decoder
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)] # a list of decoder layers
        self.dropout = krs.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask) # enc_output is fed into it

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
        return x, attention_weights
```


### Final model

#### Code Block #29: Define a class for the Transformer object

If we refer to the picture of the transformer's architecture again, we can see all the final parts it consists of. 

Here, we just add those parts together to form a transformer object. It has multi-layered encoder and decoder blocks, and a final dense layer to normalize the output.

In a call method, we first get the encoder output, then we feed it into the decoder to get the final attention weights and then we normalize them with a dense layer.
```
class Transformer(krs.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                                     target_vocab_size, pe_input, pe_target, rate=d_out_rate):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = krs.layers.Dense(target_vocab_size, kernel_regularizer=krs.regularizers.l2(l=lmbda_l2))
        
        
    # training argument is used in dropout inputs
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
       
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
```


#### Code Block #30: Passing in the hyperparameters

Here, we simply put all the defined hyperparameters into the argument space for our transformer object.
```
transformer = Transformer(
    num_layers, 
    d_model, 
    num_heads, 
    dff,
    encoder_vocab_size, 
    decoder_vocab_size, 
    pe_input=max_len_news,
    pe_target=max_len_summary,
)
```


### Masking

#### Code Block #31: Add padding and look-ahead masks

In the previous blocks of code, we defined introduced padding to some vectors to either get the same vector size or replace unknown word vectors (that we couldn’t find in a pre-trained word embeddings file). 

Now, we need to create a padding mask function to consider already existing 0 padding in our input data, so it does not affect the loss.

We also define a look-ahead mask, which simply replaces some portions at the end of our sequences to introduce learning. If we don't do that, we risk introducing cheating to our model, which means that it will know the future values it tries to predict. (We don't want that).
```
# Padding mask for masking "pad" sequences so 
# they won't affect the loss
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

# Lookahead mask for masking future words from
# contributing in prediction of current words in self attention
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
```

#### Code Block #32: Make a single function for masks to use in training 

We defined two masking functions in the previous step so now we use them in a single function. 

This function creates padding masks for both the encoder and decoder. Then, it creates a combined look-ahead and padding mask for the decoder, because this is the block, we want our model to learn from.
```
# this function is use in training step
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
        
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, combined_mask, dec_padding_mask
```


### Training

#### Code Block #33: Add exponential decay to the training schedule

We want to add exponential decay to our training so that the learning rate will decrease over time. We set the starting learning rate as defined in our hyperparameters step. The decay rate is how fast the learning rate will be decreasing after a certain number of epochs.
```
lr_schedule = krs.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr, # originally was 1e-5
    decay_steps=4000, # approximately 5 epochs
    decay_rate=0.95) # originally was 0.9
```

#### Code Block #33: Define optimizer and loss object

In this step, we add an optimizer and a loss function to our model. The optimizer is responsible for the weights’ updates during the validation step, and the loss function is used to evaluate the output and keep track of model training performance.
```
optimizer2 = Adam(lr_schedule , beta_1=0.9, beta_2=0.98, epsilon=1e-9) # changed to init
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none') # added softmax changed from_logits to false
```

#### Code Block #34: Create a loss function

To use the loss metric during our training phase, we need to implement it as a function. We will have a switch for L2 regularization, because when we want to compute the loss of the data after the regularization step. During regularization, we simply compute and add the L2 value to the final loss to account for any overfitting cases.
```
def loss_function(real, pred, l2= False):
 
    if l2:
        lambda_ = 0.0001
        l2_norms = [tf.nn.l2_loss(v) for v in transformer.trainable_variables]
        l2_norm = tf.reduce_sum(l2_norms)
        l2_value = lambda_ * l2_norm
        loss_ = loss_object(real, pred) + l2_value
    else:
        loss_ = loss_object(real, pred) 
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
```

#### Code Block #35: Add checkpoints for training

During long and complicated training processes, it may be useful to use checkpoints to save the train parameters over time. Here, we define a checkpoint manager that creates 100 checkpoints, which we can retrieve, throughout our training process.
```
checkpoint_path4 ="checkpoints4"

ckpt4 = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer2)

ckpt_manager4 = tf.train.CheckpointManager(ckpt4, checkpoint_path4, max_to_keep=100)

# if ckpt_manager4.latest_checkpoint:
#     ckpt4.restore(ckpt_manager4.latest_checkpoint)
#     print ('Latest checkpoint restored!!')
```


### Inference

#### Code Block #36: Evaluation and Summary (final) functions

In the evaluation step, our goal is to run the test data through our transformer object and generate predicted output and weights. 

We first create sequences of tokens for each sentence from the input document and add padding. Then, we create inputs for embedding inputs for the encoder and decoder, as well as define the dimensions for the output.

Lastly, in the for-loop, we create padding and look-ahead masks and feed our new data into a transformer object to generate predictions and final attention weights. The loop is the same size as the maximum length summary we defined in the early steps (retrieved from the 90th percentile of all summaries). After we reach that maximum, we stop training and concat our results.

The summary function at the end is used to simply run the whole model on any input test data and give the output in a decoded format (text format). 
```
def evaluate(input_document):
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = krs.preprocessing.sequence.pad_sequences(input_document, maxlen=max_len_news, 
                                                                           padding='post', truncating='post')
    
    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [summary_tokenizer.word_index[start_token]]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(max_len_summary):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # stop prediciting if it reached end_token
        if predicted_id == summary_tokenizer.word_index[end_token]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights

def summarize(input_document):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)  # remove start_token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document
```

#### Code Block #37: Validation function

Here, we simply run the model on the validation dataset we defined in the first steps. We split into input test data and real data to compare the two after. We again create padding and look-ahead masks and feed validation test data into our transformer. 

After getting the predictions of summaries, we calculate the validation loss with loss_function(), which we use later on when we graph the performance of our model.
```
def validate():
    print('validation started ...')
    val_loss.reset_states()
    for (batch, (inp, tar)) in enumerate(dataset_val):    
        tar_inp = tar[:, :-1] # <startseq> hi im moein
        tar_real = tar[:, 1:] # hi im moein <endseq>

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        # Operations are recorded if they are executed within this context manager
        # and at least one of their inputs is being "watched". Trainable variables are automatically watched
        predictions, _ = transformer(
            inp, tar_inp, 
            False, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)
        val_loss(loss)
    print('\n* Validation loss: {} '.format(val_loss.result()) )
    return val_loss.result()
# validate()
```

#### Code Block #38: Define a training step

In a single training step, we want to split the data into train and real data, create all the masks, and finally feed the data into our transformer.

Here, in the for-loop, we also use GradientTape() function. It allows us to keep track of gradients and access them in our training process. Since we defined our optimizer, we will use it on resulting gradients to tune our model. Finally, after each training step, we want to compute the training loss, which is essentially the mean loss after training.
```
@tf.function # Compiles a function into a callable TensorFlow graph
def train_step(inp, tar):
    tar_inp = tar[:, :-1] # <startseq> hi im moein
    tar_real = tar[:, 1:] # hi im moein <endseq>

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    # Operations are recorded if they are executed within this context manager
    # and at least one of their inputs is being "watched". Trainable variables are automatically watched
    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp, 
            True, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer2.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    # mean the loss with new computed  loss of the step
    train_loss(loss)
```

#### Code Block #39: Define the number of epochs and create loss variables

In this step, we simply define how many epochs our model will go through as well as define mean train and validation loss variables to use in a graph later.
```
history={'val':[],'train':[]}
EPOCHS = 300
not_progressing = 0
# Computes the (weighted) mean of the given loss values.
train_loss = krs.metrics.Mean(name='train_loss')
val_loss = krs.metrics.Mean(name='val_loss')
```

#### Code Block #40: Check all the input parameters

We check all the input parameter values that we will pass into the model.
```
params = {
'lmbda_l2' : lmbda_l2,
'd_out_rate' :d_out_rate,
'num_layers' : num_layers ,
'd_model' : d_model  ,
'dff' : dff ,
'num_heads' : num_heads,
'init_lr':init_lr}
params
```
```
{'lmbda_l2': 0.1,
 'd_out_rate': 0.1,
 'num_layers': 4,
 'd_model': 50,
 'dff': 512,
 'num_heads': 5,
 'init_lr': 0.001}
```

#### Code Block #41: Set random validation sets

For our model, we randomly split our validation summary set into 4 validation datasets that we will test our final model on.
```
ep = 1
best_val_loss = np.inf
i1,i2,i3,i4 = np.random.randint(len(summaries_val)),np.random.randint(len(summaries_val)),np.random.randint(len(summaries_val)),np.random.randint(len(summaries_val))
```

#### Code Block #42: Running our model and creating graphs

This code block trains our model in batches for a defined number of epochs. During each epoch, we make predictions on the validation sets and keep track of our validation loss on the graph. We also set the validation loss threshold and save certain epochs' steps. (This is the most time-consuming code block)
```
print(params)
print('#'*40)

for epoch in range(ep,EPOCHS+1):
    ep = epoch
    start = time.time()

    train_loss.reset_states()
  
    for (batch, (inp, tar)) in enumerate(dataset):
        
        train_step(inp, tar)
    
        if batch % 150 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch , batch, train_loss.result()))
                  
    print()
    print(summarize(clean_words(longreview_val['long'][i1])))
    print(summarize(clean_words(longreview_val['long'][i2])))
    print(summarize(clean_words(longreview_val['long'][i3])))
    print(summarize(clean_words(longreview_val['long'][i4])))
    print()
    
    val_loss_ = validate().numpy()
    history['val'].append((epoch,val_loss_))
    print ('\n* Train Loss {:.4f}'.format(train_loss.result()))
    history['train'].append((epoch,train_loss.result().numpy()))
    
    
    if best_val_loss-val_loss_ > 0.1:
        ckpt_save_path4 = ckpt_manager4.save()
        print ('\nSaving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path4))  
        best_val_loss = val_loss_
    
    hist(history)
    print('Current Lr: ',optimizer2._decayed_lr('float32').numpy())
    print ('\nTime taken for this epoch: {:.2f} secs\n'.format(time.time() - start))
    print('='*40)
```

#### Code Block #43: Create the time-series graph for validation loss 

Here, we simply output the concated graph that shows every validation loss for every epoch (training step) of our model. We can clearly see that the final model converges and the loss dynamic on validation set is similar to the training set.
```
hist(history)
params
```

#### Code Block #44: Testing the model on custom text

In these last two code blocks, we manually select full texts from our validation set and try running them through our model to observe the output in a text format (not as a validation loss).

We can see that the algorithm seems to summarize the text pretty well and create a meaningful and grammatically correct short sentence as a summary.
```
print(clean_words(longreview_val['long'][i1]))
print()
print(summarize(clean_words(longreview_val['long'][i1])))
```
```
a tenminute video traces us presidential elections in order to provide a historical insight into the event the video discusses the elections held less than a year after the assassination of the th us president john f kennedy further detailing the history it describes the elections wherein barack obama was voted as the th us president video explains the us elections
```
```
print(clean_words(longreview_val['long'][i2]))
print()
print(summarize(clean_words(longreview_val['long'][i2])))
```
```
brazilian police on wednesday arrested the head of the european olympic committees patrick hickey in rio de janeiro over illegal sales of olympic tickets police said hickey and at least six others are accused of illegally passing on tickets for the games to be sold on at extortionate prices hickey was taken to hospital after his arrest brazil police arrests rio olympics officials
```


