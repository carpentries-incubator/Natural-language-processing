---
title: 'From words to vectors'
teaching: 60
exercises: 60
---

::: questions
-   Why do we need to preprocess text in NLP?
-   What are the most common preprocessing operations and in which contexts should each be used?
-   What properties do word embeddings have?
-   What is a word2vec model?
-   What insights can I get from word embeddings?
-   How do I train my own word2vec model?

:::

::: objectives
After following this lesson, learners will be able to:

-   Perform basic NLP preprocessing operations
-   Implement a basic NLP Pipeline
-   Explain the motivation for vectorization in modern NLP
-   Train a custom Word2Vec model using the [Gensim](https://radimrehurek.com/gensim/) library
-   Apply a Word2Vec model to interpret and analyze semantics of text (either a pre-trained model or custom model)
-   Describe the kinds of semantic relationships captured by Word2Vec, and identify NLP tasks it is suited to support
-   Explain, with examples, what the limitations are for the Word2Vec representation
:::

## Introduction

In this episode, we will learn about the importance of preprocessing text in NLP, and how to apply common preprocessing operations to text files. We will also learn more about *NLP Pipelines*, learn about their basic components and how to construct such pipelines.

We will then address the transition from rule-based NLP to distributional semantics approaches which encode text into numerical representations based on statistical relationships between tokens. We will introduce one particular algorithm for this kind of encoding called Word2Vec proposed in 2013 by [Mikolov et al](https://arxiv.org/pdf/1301.3781). We will show what kind of useful semantic relationships these representations encode in text, and how we can use them to solve specific NLP tasks. We will also discuss some of the limitations of Word2Vec which are addressed in the next lesson on transformers before concluding with a summary of what we covered in this lesson.

## Preprocessing Operations

As is common in data science and machine learning, raw textual data often comes in a form that is not readily suitable for downstream NLP tasks. Preprocessing operations in NLP are analogous to the data cleaning and sanitation steps in traditional non-NLP Machine Learning tasks. Sometimes you are extracting text from PDF files which contain line breaks, headers, tables etc. that are not relevant to NLP tasks and which need to be removed. You may need to remove punctuation and special characters, or lowercase text for some NLP tasks etc. Whether you need to perform certain preprocessing operations, and the order in which you should perform them, will depend on the NLP task at hand.

Also note that preprocessing can differ significantly if you work with different languages. This is both in terms of which steps to apply, but also which methods to use for a specific step.

Here we will analyze with more detail the most common pre-processing steps when dealing with unstructured English text data:

### Data Formatting
Text comes from various sources and are available in different formats (e.g., Microsoft Word documents, PDF documents, ePub files, plain text files, Web pages etc...). The first step is to obtain a clean text representation that can be transferred into python UTF-8 strings that our scripts can manipulate.

Take a look at the `data/84_frankenstein_or_the_modern_prometheus.txt` file: 

```python
filename = "data/84_frankenstein_or_the_modern_prometheus.txt"
with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()

print(text[:300]) # print the first 300 characters
```

Our file is already in plain text so it might seem we do not need to do anything; however, if we look closer we see new line characters separating not only paragraphs but breaking the lines in the middle of sentences. While this is useful to keep the text in a narrow space to help the human reader, it introduces artificial breaks that can confuse any automatic analysis (for example to identify where sentences start and end). 

One straightforward way is to replace the new lines with spaces so all the text is in a single line:

```python
text_flat = text.replace("\n", " ")
print(text_flat[:300]) # print the first 300 characters
```

Other data formatting operations might include:
- Removal of special or noisy characters. For example:

    - Random symbols: "The total cost is $120.00#" → remove #
    - Incorrectly recognized letters or numbers: 1 misread as l, 0 as O, etc. Example: "l0ve" → should be "love"
    - Control or formatting characters: \n, \t, \r appearing in the middle of sentences. Example: "Please\nsubmit\tyour form." → "Please submit your form."
    - Non-standard Unicode characters: �, �, or other placeholder symbols where OCR failed. Example: "Th� quick brown fox" → "The quick brown fox"
- Remove HTML tags (e.g., if you are extracting text from Web pages)
- Strip non-meaningful punctuation (e.g., "The quick brown fox jumps over the lazy dog and con-
tinues to run across the field.)
- Strip footnotes, headers, tables, images etc.
- Remove URLs or phone numbers

::: callout
What if I need to extract text from MS Word docs or PDF files or Web pages There are various Python libraries for helping you extract and manipulate text from these kinds of sources.

- For MS Word documents [python-docx](https://python-docx.readthedocs.io/en/latest/) is popular.
- For (text-based) PDF files [PyPDF2](https://pypi.org/project/PyPDF2/) and [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) are widely used. Note that some PDF files are encoded as images (pixels) and not text. If the text in these files is digital (as opposed to scanned handwriting), you can use OCR (Optical Character Recognition) libraries such as [pytesseract](https://pypi.org/project/pytesseract/) to convert the image to machine-readable text.
- For scraping text from websites, [BeautifulSoup](https://pypi.org/project/beautifulsoup4/) and [Scrapy](https://docs.scrapy.org/en/latest/) are some common options.
- LLMs also have something to offer here, and the field is moving pretty fast. There are some interesting open source LLM-based document parsers and OCR-like extractors such as [Marker](https://github.com/datalab-to/marker), or [PyMuPDF4LLM](https://github.com/pymupdf/PyMuPDF4LLM), just to mention a couple.
:::


::: callout
Another important choice at the data formatting level is to decide at what granularity do you need to perform the NLP task: 

- Are you analyzing phenomena at the **word level**? For example, detecting abusive language (based on a known vocabulary).
- Do you need to first extract sentences from the text and do analysis at the **sentence level**? For example, extracting entities in each sentence.
- Do you need full **chunks of text**? (e.g. paragraphs or chapters?) For example, summarizing each paragraph in a document.
- Or perhaps you want to extract patterns at the **document level**? For example each full book should have one genre tag (Romance, History, Poetry).

Sometimes your data will be already available at the desired granularity level. If this is not the case, then during the tokenization step you will need to figure out how to obtain the desired granularity level.
:::

### Tokenization 

Tokenization is a foundational operation in NLP, as it helps to create structure from raw text. This structure is a basic requirement and input for modern NLP algorithms to attribute and interpret meaning from text. This operation involves the segmentation of the text into smaller units referred to as `tokens`. Tokens can be sentences (e.g. `'the happy cat'`), words (`'the', 'happy', 'cat'`), subwords (`'un', 'happiness'`) or characters (`'c','a', 't'`). Different NLP algorithms may require different choices for the token unit. And different languages may require different approaches to identify or segment these tokens.

Python strings are by definition sequences of characters, thus we can iterate through the string character by character:

```python
print(type(text_flat))  # Should be <class 'str'>
for ch in text_flat:
    print(ch)
```

 However, it is often more advantageous if our atomic units are words. As we saw in Lesson 1, the task of extracting word tokens from texts is not trivial, therefore pre-trained models such as spaCy can help with this step. In this case we will use the small English model that was trained on a web corpus:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)
print(type(doc))  # Should be <class 'spacy.tokens.doc.Doc'>
print(len(doc))
print(doc)
```

::: callout
A good word tokenizer for example, does not simply break up a text based on spaces and punctuation, but it should be able to distinguish:

-   abbreviations that include points (e.g.: *e.g.*)
-   times (*11:15*) and dates written in various formats (*01/01/2024* or *01-01-2024*)
-   word contractions such as *don't*, these should be split into *do* and *n't*
-   URLs

Many older tokenizers are rule-based, meaning that they iterate over a number of predefined rules to split the text into tokens, which is useful for splitting text into word tokens for example. Modern large language models use subword tokenization, which learn to break text into pieces that are statistically convenient, this makes them more flexible but less human-readable.
:::

We can access the tokens by iterating the document and getting its `.text` property:

``` python
tokens_txt = [token.text for token in doc]
print(tokens_txt[:15])
```

``` output
['Letter', '1', '\n\n\n', 'St.', 'Petersburgh', ',', 'Dec.', '11th', ',', '17', '-', '-', '\n\n', 'TO', 'Mrs.']
```

This shows us the individual tokens, including new lines and punctuation (in case we didn't run the previous cleaning step). spaCy allows us to filter based on token properties. For example, assuming we are not interested in the newlines, punctuation nor in numeric tokens, in one single step we can keep only the token objects that contain alphabetical characters:

``` python
tokens = [token for token in doc if token.is_alpha]
print(tokens[:15])
```

``` output
[Letter, Petersburgh, TO, Saville, England, You, will, rejoice, to, hear, that, no, disaster, has, accompanied]
```

We do not have to depend necessarily on the `Doc` and `Token` spaCy objects. Once we tokenized the text with the spaCy model, we can extract the list of words as a list of strings and continue our text analysis:

```python
words = [token.text for token in doc]
print(words[:20])
```

Again, it all depends on what your requirements are. For example, sometimes it is more useful if our atomic units are sentences. Think of the NLP task of classifying each whole sentence inside a text as Positive/Negative/Neutral in terms of sentiment (e.g., within movie reviews). spaCy also helps with this using a sentence segmentation model:

```python
sentences = [sent.text for sent in doc.sents]
[print(s) for s in sentences[:5]]
```

Note that in this case each sentence is a python object, and the property `.text` returns an untokenized string (in terms of words). But we can still access the list of word tokens inside each sentence object if we want:

```python
sents_sample = list(doc.sents)[:10]
for sent in sents_sample:
    print("Sentence:", sent.text)
    for token in sent:
        print("\tToken:", token.text)
```

This will give us enough flexibility to work at the sentence and word level at the same time. In terms of what we can do with these sentences once spaCy has identified them, we could ask humans to label each sentence as either Positive/Negative/Neutral and train a supervised model for sentiment classification on the set of sentences. Or if we have a pre-trained model for sentiment classification on sentences, we could load this model and classify each of our input sentences as either Positive/Negative/Neutral.

### Lowercasing

Removing uppercases to e.g. avoid treating "Dog" and "dog" as two different words is also a common step, for example to train word vector representations, we want to merge both occurrences as they represent exactly the same concept. Lowercasing can be done with Python directly as:

```python
lower_text = text_flat.lower()
lower_text[:100] # Beware that this is a python string operation
```

Beware that lowercasing the whole string as a first step might affect the tokenizer behavior since tokenization benefits from information provided by case-sensitive strings. We can therefore tokenize first using spaCy and then obtain the lowercase strings of each token using the `.lower_` property:

```python
lower_text = [token.lower_ for token in doc]
lower_text[:10] # Beware that this is a list of strings now!
```

In other tasks, such as Named Entity Recognition (NER), lowercasing before training can actually lower the performance of your model. This is because words that start with an uppercase (not preceded by a period) can represent proper nouns that map into Entities, for example:

```python
import spacy
# Preserving uppercase characters increases the likelihood that an NER model
# will correctly identify Apple and Will as a company (ORG) and a person (PER)
# respectively.
str1 = "My next laptop will be from Apple, Will said." 
# Lowercasing can reduce the likelihood of accurate labeling
str2 = "my next laptop will be from apple, will said."

nlp = spacy.load("en_core_web_sm")
ents1 = [ent.text for ent in nlp(str1).ents]
ents2 = [ent.text for ent in nlp(str2).ents]

print(ents1)
print(ents2)
```

```output
['Apple', 'Will']
[]
```

### Lemmatization 
Although it has become less widely used in modern NLP approaches, normalizing words into their *dictionary form* can help to focus on relevant aspects of text. Consider how "eating", "ate", "eaten" are all variations of the root verb "eat". Each variation is sometimes known as an _inflection_ of the root word. Conversely, we say that the word "eat" is the _lemma_ for the words "eating", "eats", "eaten", "ate" etc. Lemmatization is therefore the process of rewriting each token or word in a given input text as its lemma.

Lemmatization is not only a possible preprocessing step in NLP but also an NLP task on its own, with different algorithms for it. Therefore we also tend to use pre-trained models to perform lemmatization. Using spaCy we can access the lemmmatized version of each token with the `lemma_` property (notice the underscore!):

```python
lemmas = [token.lemma_ for token in doc]
print(lemmas[:50])
```

Note that the list of lemmas is now a list of strings.

Having a lemmatized text allows us to merge the different surface occurrences of the same concept into a single token. This can be very useful for count-based NLP methods such as topic modelling approaches which count the frequency of certain words to see how prevalent a given topic is within a document. If you condense "eat", "eating", "ate", "eaten" to the same token "eat" then you can count four occurrences of the same "topic" in a text, instead of treating these four tokens as distinct or unrelated topics just because they are spelled differently. You can also use lemmatization for generating word embeddings. For example, you can have a single vector for `eat` instead of one vector per verb tense. 

As with each preprocessing operation, this step is optional. Consider, for example, the cases where the differences of verb usage according to tense is informative, or the difference between singular and plural usage of nouns, in those cases lemmatizing will get rid of important information for your task. For example, if your chosen NLP task is to detect past tense verbs from a document, then lemmatizing "eaten" into "eat" loses crucial information about tense that your model requires.

### Stop Word Removal
The most frequent words in texts are those which contribute little semantic value on their own: articles ('the', 'a', 'an'), conjunctions ('and', 'or', 'but'), prepositions ('on', 'by'), auxiliary verbs ('is', 'am'), pronouns ('he', 'which'), or any highly frequent word that might not be of interest in several *content-only* related tasks. Let's define a small list of stop words for this specific case:

```python
STOP_WORDS = ["the", "you", "will"] # This list can be customized to your needs...
```

Using Python directly, we need to manually define a list of what we consider to be stop words and directly filter the tokens that match this. Notice that lemmatization was a crucial step to get more coverage with the stop word filtering:

```python
lemmas = [token.lemma_ for token in doc]
content_words = []
for lemma in lemmas:
    if lemma not in STOP_WORDS:
        content_words.append(lemma)
print(content_words[:20])
```

Using spaCy we can filter the stop words based on the token properties:

``` python
tokens_nostop = [token for token in tokens if not token.is_stop]
print(tokens[:15])
```

There is no canonical definition of stop words because what you consider to be a stop word is directly linked to the objective of your task at hand. For example, pronouns are usually considered stopwords, but if you want to do gender bias analysis then pronouns are actually a key element of your text processing pipeline. Similarly, removing articles and prepositions from text is obviously not advised if you are doing _dependency parsing_ (the task of identifying the parts of speech in a given text).

Another special case is the word 'not' which may encode the semantic notion of _negation_. Removing such tokens can drastically change the meaning of sentences and therefore affect the accuracy of models for which negation is important to preserve (e.g., sentiment classification "this movie was NOT great" vs. "this movie was great").

## NLP Pipeline

The concept of NLP pipeline refers to the sequence of operations that we apply to our data in order to go from the original data (e.g. original raw documents) to the expected outputs of our NLP Task at hand. The components of the pipeline refer to any manipulation we apply to the text, and do not necessarily need to be complex models, they involve preprocessing operations, application of rules or machine learning models, as well as formatting the outputs in a desired way.

### A simple rule-based classifier
Imagine we want to build a very lightweight sentiment classifier. A basic approach is to design the following pipeline: 

1. Clean the original text file (as we saw in the Data Formatting section)
2. Apply a sentence segmentation or tokenization model
3. Define a set of positive and negative words (a hard coded dictionary)
4. For each sentence: 
    - If it contains one or more of the positive words, classify as `POSITIVE`
    - If it contains one or more of the negative words, classify as `NEGATIVE`
    - Otherwise classify as `NEUTRAL`
5. Output a table with the original sentence and the assigned label


This is implemented with the following code:

1. Read the text and normalize it into a single line

```python
import spacy
nlp = spacy.load("en_core_web_sm")

filename = "data/84_frankenstein_or_the_modern_prometheus.txt"
with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()

text = text.replace("\n", " ") # some cleaning by removing new line characters
```

2. Apply Sentence segmentation

```python
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
```

3. Define the positive and negative words you care about:

```python
positive_words = ["happy", "excited", "delighted", "content", "love", "enjoyment"]
negative_words = ["unhappy", "sad", "anxious", "miserable", "fear", "horror"]
```

4. Apply the rules to each sentence and collect the labels

```python
classified_sentences = []

for sent in sentences:
    if any(word in sent.lower() for word in positive_words):
        classified_sentences.append((sent, 'POSITIVE'))
    elif any(word in sent.lower() for word in negative_words):
        classified_sentences.append((sent, 'NEGATIVE'))
    else:
        classified_sentences.append((sent, 'NEUTRAL'))
```

5. Save the classified data

```python
import pandas as pd
df = pd.DataFrame(classified_sentences, columns=['sentence', 'label'])
df.to_csv('results_naive_rule_classifier.csv', sep='\t')
```

:::: challenge
Discuss the pros and cons of the proposed NLP pipeline: 

1. Do you think it will give accurate results?
2. What do you think about the coverage of this approach? What cases will it miss? 
3. Think of possible drawbacks of chaining components in a pipeline.

::: solution

1. This classifier only considers the presence of one word to apply a label. It does not analyze sentence semantics or even syntax.
2. Given how the rules are defined, if both positive and negative words are present in the same sentence it will assign the `POSITIVE` label. It will generate a lot of false positives because of the simplistic rules
3. The errors from previous steps get carried over to the next steps increasing the likelihood of noisy outputs.
:::

::::

So far we’ve seen how to format and segment the text to have atomic data at the word level or sentence level. We then apply operations to the word and sentence strings. This approach still depends on counting and exact keyword matching. And as we have already seen it has several limitations. The method cannot interpret words outside the dictionary defined for example.

One way to combat this is by transforming each word into numeric representation and study statistical patterns in how these words are distributed in text. For example, what words tend to occur "close" to a given word in my data? For example, if we analyze restaurant menus we find that "cheese", "mozzarella", "base" etc. frequently occur near the token "pizza". We can then exploit these statistical patterns to inform various NLP tasks. This concept is commonly known as [distributional semantics](https://arxiv.org/pdf/1905.01896). It is based on the assumption "words that appear in similar contexts have similar meanings.”

This concept is powerful for enabling, for example, the measurement of semantic similarity of words, sentences, phrases etc. in text. And this, in turn, can help with other downstream NLP tasks, as we shall see in the next section on word embeddings.

## Word Embeddings

### Reminder: Neural Networks

Understanding how neural networks work is out of the scope of this course. For our purposes we will simplify the explanation in order to conceptually understand how Neural Network works. A Neural Network (NN) is a pattern-finding machine with layers (a *deep* neural network is the same concept but scaled to dozens or even hundreds of layers). In a neural network, each layer has several interconnected *neurons*, each one corresponding to a random number initially. The deeper the network is, the more complex patterns it can learn. As the neural netork gets trained (that is, as it sees several labeled examples that we provide), each neuron value will be updated in order to maximize the probability of getting the answers right. A well trained neural network will be able to predict the right labels on completely new data with certain accuracy. 

![After seeing thousands of examples, each layer represents different “features” that maximize the success of the task, but they are not human-readable. The last layer acts as a classifier and outputs the most likely label given the input](fig/emb_neuralnet.png)

The main difference with traditional machine learning models is that we do not need to design explicitly any features, rather the network will *adjust itself* by looking at the data alone and executing the back-propagation algorithm. The main job when using NNs is to encode our data properly so it can be fed into the network. 

### Rationale behind Embeddings

**A word embedding is a numeric vector that represents a word**. Word2Vec exploits the "feature agnostic" power of neural networks to transform word strings into trained word numeric representations. Hence we still use words as features but instead of using the string directly, we transform that string into its corresponding vector in the pre-trained Word2Vec model. And because both the network input and output are the words themselves in text, we basically have billions of *labeled* training datapoints for free.

![](fig/emb_embeddings.png)

To obtained the word embeddings, a shallow neural network is optimized with the task of language modeling and the final hidden layer inside the trained network holds the fixed size vectors whose values can be mapped into linguistic properties (since the training objective was language modeling). Since similar words occur in similar contexts, or have same characteristics, a properly trained model will learn to assign similar vectors to similar words.

By representing words with vectors, we can mathematically manipulate them through vector arithmetic and express semantic similarity in terms of vector distance. Because the size of the learned vectors is not proportional to the amount of documents we can learn the representations from larger collections of texts, obtaining more robust representations, that are less corpus-dependent.


There are two main algorithms for training Word2Vec:

-   Continuous Bag-of-Words (CBOW): Predicts a target word based on its surrounding context words.
-   Continuous Skip-Gram: Predicts surrounding context words given a target word.

![](fig/emb13.png)

If you want to know more about the technicl aspecs of training Word2Vec you can visit this [tutorial](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

### The Word2Vec Vector Space

The python module `gensim` offers a user-friendly interface to interact with pre-trained Word2vec models and also to train our own. First we will explore the model from the original Word2Vec paper, which was trained on a big corpus from Google News (English news articles). We will see what functionalities are available to explore a vector space. Then we will prepare our own text step-by-step to train our own Word2vec models and save them.

### Load the embeddings and inspect them

The library `gensim` has a repository with English pre-trained models. We can take a look at the models:

``` python
import gensim.downloader
available_models = gensim.downloader.info()['models'].keys()
print(list(available_models))
```

We will download the google News model with:

``` python
w2v_model = gensim.downloader.load('word2vec-google-news-300')
```

We can do some basic checkups such as showing how many words are in the vocabulary (i.e., for how many words do we have an available vector), what is the total number of dimensions in each vector, and print the components of a vector for a given word:

``` python
print(len(w2v_model.key_to_index.keys())) # 3 million words
print(w2v_model.vector_size) # 300 dimensions. This can be chosen when training your own model
print(w2v_model['car'][:10]) # The first 10 dimensions of the vector representing 'car'.
print(w2v_model['cat'][:10]) # The first 10 dimensions of the vector representing 'cat'.
```

``` output
3000000
300
[ 0.13085938  0.00842285  0.03344727 -0.05883789  0.04003906 -0.14257812
  0.04931641 -0.16894531  0.20898438  0.11962891]
[ 0.0123291   0.20410156 -0.28515625  0.21679688  0.11816406  0.08300781
  0.04980469 -0.00952148  0.22070312 -0.12597656]
```

As we can see, this is a very large model with 3 million words and the dimensionality chosen at training time was 300, thus each word will have a 300-dimension vector associated with it.

Even with such a big vocabulary we can always find a word that won't be in there:

``` python
print(w2v_model['bazzinga'][:10])
```

This will throw a `KeyError` as the model does not know that word. Unfortunately this is a limitation of Word2vec - unseen words (words that were not included in the training data) cannot be interpreted by the model. 

Now let's talk about the vectors themselves. They are not easy to interpret as they are a bunch of floating point numbers. These are the weights that the network learned when optimizing for language modelling. As the vectors are hard to interpret, we rely on a mathematical method to compute how similar two vectors are. Generally speaking, the recommended metric for measuring similarity between two high-dimensional vectors is [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) .

::: callout
[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) ranges between [`-1` and `1`]. It is the cosine of the angle between two vectors, divided by the product of their length. Mathematically speaking, when two vectors point in exactly the same direction their cosine will be 1, and when they point in the opposite direction their cosine will be -1. In python we can use Numpy to compute the cosine similarity of vectors.

![](fig/emb12.png){alt=""}
:::

We can use `sklearn` learn to measure any pair of high-dimensional vectors:

``` python
from sklearn.metrics.pairwise import cosine_similarity

car_vector = w2v_model['car']
cat_vector = w2v_model['cat']

similarity = cosine_similarity([car_vector], [cat_vector])
print(f"Cosine similarity between 'car' and 'cat': {similarity[0][0]}")

similarity = cosine_similarity([w2v_model['hamburger']], [w2v_model['pizza']])
print(f"Cosine similarity between 'hamburger' and 'pizza': {similarity[0][0]}")
```

``` python

Cosine similarity between 'car' and 'cat': 0.21528185904026031
Cosine similarity between 'hamburger' and 'pizza': 0.6153676509857178
```

Or you can use directly the `w2v_model.similarity('car', 'cat')` function which gives the same result.

The higher similarity score between the hamburger and pizza indicates they are more similar based on the contexts where they appear in the training data. Even though is hard to read all the floating numbers in the vectors, we can trust this metric to always give us a hint of which words are semantically closer than others

:::: challenge
Think of different word pairs and try to guess how close or distant they will be from each other. Use the similarity measure from the word2vec module to compute the metric and discuss if this fits your expectations. If not, can you come up with a reason why this was not the case?

::: solution
Some interesting cases include synonyms, antonyms and morphologically related words:

``` python
print(w2v_model.similarity('democracy', 'democratic'))
print(w2v_model.similarity('queen', 'princess'))
print(w2v_model.similarity('love', 'hate')) #!! (think of "I love X" and "I hate X")
print(w2v_model.similarity('love', 'lover'))
```

``` output
0.86444813
0.7070532
0.6003957
0.48608577
```
:::
::::

### Vector Neighborhoods

Now that we have a metric we can trust, we can retrieve neighborhoods of vectors that are close to a given word. This is analogous to retrieving semantically related terms to a target term. Let's explore the neighborhood around \`pizza\` using the `most_similar()` method:

``` python
print(w2v_model.most_similar('pizza', topn=10))
```

This returns a list of ranked tuples with the form (word, similarity_score). The list is already ordered in descent, so the first element is the closest vector in the vector space, the second element is the second closest word and so on...

``` output
[('pizzas', 0.7863470911979675), 
('Domino_pizza', 0.7342829704284668), 
('Pizza', 0.6988078355789185), 
('pepperoni_pizza', 0.6902607083320618), 
('sandwich', 0.6840401887893677), 
('burger', 0.6569692492485046), 
('sandwiches', 0.6495091319084167), 
('takeout_pizza', 0.6491535902023315), 
('gourmet_pizza', 0.6400628089904785), 
('meatball_sandwich', 0.6377009749412537)]
```

Exploring neighborhoods can help us understand why some vectors are closer (or not so much). Take the case of *love* and *lover*, originally we might think these should be very close to each other but by looking at their neighborhoods we understand why this is not the case:

``` python
print(w2v_model.most_similar('love', topn=10))
print(w2v_model.most_similar('lover', topn=10))
```

This returns a list of ranked tuples with the form (word, similarity_score). The list is already ordered in descent, so the first element is the closest vector in the vector space, the second element is the second closest word and so on...

``` output
[('loved', 0.6907791495323181), ('adore', 0.6816874146461487), ('loves', 0.6618633270263672), ('passion', 0.6100709438323975), ('hate', 0.6003956198692322), ('loving', 0.5886634588241577), ('Ilove', 0.5702950954437256), ('affection', 0.5664337873458862), ('undying_love', 0.5547305345535278), ('absolutely_adore', 0.5536840558052063)]

[('paramour', 0.6798686385154724), ('mistress', 0.6387110352516174), ('boyfriend', 0.6375402212142944), ('lovers', 0.6339589953422546), ('girlfriend', 0.6140860915184021), ('beau', 0.609399676322937), ('fiancé', 0.5994566679000854), ('soulmate', 0.5993717312812805), ('hubby', 0.5904166102409363), ('fiancée', 0.5888950228691101)]
```

The first word is a noun or a verb (depending on the context) that denotes affection to someone/something , so it is associated with other concepts of affection (positive or negative). The case of *lover* is used to describe a person, hence the associated concepts are descriptors of people with whom the lover can be associated.

### Word Analogies with Vectors

Another powerful property that word embeddings show is that vector algebra can preserve semantic analogy. An analogy is a comparison between two different things based on their similar features or relationships, for example king is to queen as man is to woman. We can mimic this operations directly on the vectors using the `most_similar()` method with the `positive` and `negative` parameters:

``` python
# king is to man as what is to woman?
# king + woman - man = queen
w2v_model.most_similar(positive=['king', 'woman'], negative=['man'])
```

``` output
[('queen', 0.7118192911148071),
 ('monarch', 0.6189674735069275),
 ('princess', 0.5902431011199951),
 ('crown_prince', 0.5499460697174072),
 ('prince', 0.5377321243286133),
 ('kings', 0.5236844420433044),
 ('Queen_Consort', 0.5235945582389832),
 ('queens', 0.5181134343147278),
 ('sultan', 0.5098593235015869),
 ('monarchy', 0.5087411403656006)]
```

## Train your own Word2Vec

The `gensim` package has implemented everything for us, this means we have to focus mostly on obtaining clean data and then calling the `Word2Vec` object to train our own model with our own data. This can be done like follows:

``` python
import spacy
from gensim.models import Word2Vec 

# Load and Tokenize the Text using spacy
spacy_model = spacy.load("en_core_web_sm")
with open("data/84_frankenstein_clean.txt") as f:
    book_text = f.read()    
book_doc = spacy_model(book_text)
clean_tokens = [tok.text.lower() for tok in book_doc if tok.is_alpha and not tok.is_stop]

# Call and Train the Word2Vec model
model = Word2Vec([clean_tokens], sg=0 , vector_size=300, window=5, min_count=1, workers=4)
```

With this line code we are configuring our whole Word2Vec training schema. We will be using CBOW (`sg=0` means CBOW, `sg=1` means Skip-gram). We are interested in having vectors with 300 dimensions `vector_size=300` and a context size of 5 surrounding words `window=5`. Because we already filtered our tokens, we include all words present in the filtered corpora, regardless of their frequency of occurrence `min_count=1`. The last parameters tells python to use 4 CPU cores for training.

See the Gensim [documentation](https://radimrehurek.com/gensim/models/word2vec.html) for more training options.

### Save and Retrieve your model

Once your model is trained it is useful to save the checkpoint in order to retrieve it next time instead of having to train it every time. You can save it with:

```python
model.save("word2vec_mini_books.model")
```

Let's put everything together. We have now the following NLP task: train our own Word2Vec model. We are interested on having vectors for content words only, so even though our preprocessing will unfortunately lose a lot of the original information, in exchange we will be able to manipulate the most relevant conceptual words as individual numeric representations. 


To load back the pre-trained vectors you just created you can use the following code:

```python
model = Word2Vec.load("word2vec_mini_books.model")
w2v = model.wv
# Test:
w2v.most_similar('monster')
```


:::: challenge
Let's apply this step by step on a longer text. In this case, because we are learning the process, our corpus will be only one book but in reality we would like to train a network with thousands of them. We will use two books: Frankenstein and Dracula to train a model of word vectors.

Write the code to follow the proposed pipeline and train the word2vec model. The proposed pipeline for this task is: 

- load the text files
- tokenize files
- keep only alphanumerical tokens
- lemmatize words
- Remove stop words
- Train a Word2Vec model (feed the clean tokens to the `Word2Vec` object) with `vector_size=50`
- Save the trained model

::: solution
```python
import spacy
from gensim.models import Word2Vec 

def process_book(book_filename: str, spacy_model: spacy.lang) -> list[str]:
    with open(book_filename) as f:
        book_text = f.read()
    
    book_doc = spacy_model(book_text)
    valid_tokens = [tok for tok in book_doc if tok.is_alpha and not tok.is_stop]
    lemmas = [tok.lemma_ for tok in valid_tokens] 
    return lemmas
    
nlp = spacy.load("en_core_web_sm")

# Load the Tokens
franken_tokens = process_book("data/84_frankenstein_clean.txt", nlp)
dracula_tokens = process_book("data/345_dracula_clean.txt", nlp)

# Train our own model
spooky_model = Word2Vec([franken_tokens, dracula_tokens], sg=0 , vector_size=50, window=5, min_count=1, workers=4)

# Test the vectors
print(len(spooky_model.wv['Frankenstein']))
print(spooky_model.wv['Frankenstein'][:30])
print(spooky_model.wv.most_similar("Frankenstein"))
```
:::

::::


::: callout
## Dataset size in training

To obtain your own high-quality embeddings, the size/length of the training dataset plays a crucial role. Generally [tens of thousands of documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) are considered a reasonable amount of data for decent results.

Is there a strict minimum? Not really. It’s important to keep in mind that ``vocabulary size``, ``document length``, and ``desired vector size`` all interact with each other. Higher-dimensional vectors (e.g., 200–300 dimensions) provide more features to capture a word’s meaning, resulting in higher-quality embeddings that can represent words across a finer-grained and more diverse set of contexts.

While Word2vec models typically perform better with large datasets containing millions of words, using a single page is sufficient for demonstration and learning purposes. This smaller dataset allows us to train the model quickly and understand how word2vec works without the need for extensive computational resources.
:::


::: keypoints
-   We can run a preprocessing pipeline to obtain clear words that can be used as features
-   We learned how are words converted into vectors of numbers (which makes them interpretable for machines)
-   We can easily compute how words are similar to each other with the cosine similarity
-   Using gensim we can train our own word2vec models
:::
