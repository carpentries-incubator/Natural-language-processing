---
title: 'Episode 2: BERT and Transformers'
teaching: 60
exercises: 60
---
:::::::::::::::::::::::::::::::::::::::::::::::: questions
- What are some drawbacks of static word embeddings?
- What are Transformers?
- What is BERT and how does it work?
- How can I use BERT to solve NLP tasks?
- How should I evaluate my classifiers?
- Which other Transformer variants are available?

:::::::::::::::::::::::::::::::::::::::::::::::: 

:::::::::::::::::::::::::::::::::::::::::::::::: objectives
- Understand how a Transformer works and recognize their different use cases.
- Understand how to use pre-trained tranfromers (Use Case: BERT)
- Use BERT to classify texts.
- Use BERT as a Named Entity Recognizer.
- Understand assumptions and basic evaluation for NLP outputs.

::::::::::::::::::::::::::::::::::::::::::::::::

Static word embeddings such as Word2Vec can be used to represent each word as a unique vectors. Vector representations also allow us to apply numerical operations that can be mapped to some syntactic and semantic properties of words, such as the cases of analogies or finding synonyms. Once we transform words into vectors, these can also be used as **features** for classifiers that can be trained predict any supervised NLP task.

However, a big drawback of Word2Vec is that **each word is represented in isolation**, and unfortunately that is not how language works. Words get their meanings based on the specific context in which they are used (take for example polysemy, the cases where the same word can have very different meanings depending on the context); therefore, we would like to have richer vector representations of words that also integrate context into account in order to obtain more powerful representations. 

:::: challenge
### Polysemy in Language

Think of (at least 2) different words that can have more than one meaning depending on the context. Come up with one simple sentence per meaning and explain what they mean in each context. Discuss: How do you know what of the possible meanings does the word have when you use it?

OPTIONAL: Why do you think Word2Vec can't caputure different meanings of words?

::: solution

Two possible examples can be the words 'fine' and 'run'

Sentences for 'fine':
- She has a fine watch (fine == high-quality)
- He had to pay a fine (fine == penalty)
- I am feeling fine (fine == not bad)

Sentences for 'run':
- I had to run to catch the bus (run == moving fast)
- Stop talking, before you run out of ideas (run (out) == exhaust)

Note how in the "run out" example we even have to understand that the meaning of run is not literal but goes accompained with a preposition that changes its meaning.

:::
::::


In 2019, the BERT language model was introduced. Using a novel architecture called Transformer (2017), BERT can integrate context into word representations. To understand BERT, we will first look at what a transformer is and we will then directly use some code to make use of BERT.

# Transformers

The Transformer is a neural network architecture proposed by Google researchers [in 2017](https://arxiv.org/pdf/1706.03762) in a paper called *Attention is all you Need*. They tackled specifically the NLP task of Machine Translation (MT), which is stated as: how to generate a sentence (sequence of words) in target language B given a sentence in source language A? We all know that translation cannot be done word by word in isolations, therefore integrating the context from both the source language and the target language is necessary. In order to translate, first one neural network needs to _encode_ the whole meaning of the senetence in language A into a single vector representation, then a second neural network needs to _decode_ that representation into tokens that are both coherent with the meaning of language A and understandable in language B. Therefore we say that translation is modeling language B _conditioned_ on what language A originally said.


![Transformer Architecture](fig/trans1.png)

As seen in the picture, the original Transformer is an Encoder-Decoder network that tackles translation. We first need a token embedder which converts the string of words into a sequence of vectors that the Transformer network can process. The first component, the __Encoder__, is optimized for creating **rich representations** of the source sequence (in this case an English sentence) while the second one, the __Decoder__ is a **generative network** that is conditioned on the encoded representation. The third component we see is the infamous attention mechanism, a third neural network what computes the correlation between source and target tokens (*Which word in Dutch should I pay attention to decide a better next English word?*) to generate the most likely token in the target sequence (in this case Dutch words). 

:::: challenge
### Emulate the Attention Mechanism

Pair with a person who speaks a language different from English (we will cal it language B). This time you should think of 2 simple sentences in English and come up with their translations in the second language. In a piece of paper write down both sentences (one on top of the other) and try to:
1. Draw a one to one mapping of words in English to language B. Is it always possible to do this?
2. Think of each word in language B and draw as many lines as necessary to the relevant English words that can "help you" predict the word in language B. If you managed, congratulations, this is how attention works!

::: solution
Here an image of a bilingual "manual attention" example
:::
::::

Next, we will see how BERT exploits the idea of a **Transformer Encoder** to perform the NLP Task we are interested in: generating powerful word representations.

# BERT

[BERT](https://aclanthology.org/N19-1423.pdf) is an acronym that stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. The name describes it all: the idea is to use the power of the Encoder component of the Transformer architecture to create powerful token representations that preserve the contextual meaning of the whole input segment, instead of each word in isolation. The BERT vector representations of each token take into account both the left context (what comes before the word) and the right context (what comes after the word). Another advantage of the transformer Encoder is that it is parallelizable, which made it posible for the first time to train these networks on millions of datapoints, dramatically improving model generalization. 

::: callout
## Pretraining BERT
To obtain the BERT vector representations the Encoder is pre-trained with two different tasks:
- **Masked Language Model:** for each sentence, mask one token at a time and predict which token is missing based on the context from both sides. A training input example would be "Maria [MASK] Groningen" and the model should predict the word "loves".
- **Next Sentence Prediction:** the Encoder gets a linear binary classifier on top, which is trained to decide for each pair of sequences A and B, if sequence A precedes sequence B in a text. For the sentence pair: "Maria loves Groningen." and "This is a city in the Netherlands." the output of the classifier is "True" and for the pair "Maria loves Groningen." and "It was a tasty cake." the output should be "false" as there is no obvious continuation between the two sentences.

Already the second pre-training task gives us an idea of the power of BERT: after it has been pretrained on hundreds of thousands of texts, one can plug-in a classifier on top and re-use the *linguistic* knowledge previously acquired to fine-tune it for a specific task, without needing to learn the weights of the whole network from scratch all over again. In the next sections we will describe the components of BERT and show how to use it. This model and hundreds of related transformer-based pre-trained encoders can also be found on [Hugging Face](https://huggingface.co/google-bert/bert-base-cased).
:::

# BERT Architecture

The BERT Architecture can be seen as a basic NLP pipeline on its own:

1. **Tokenizer:** splits text into tokens that the model recognizes
2. **Embedder:** converts each token into a fixed-sized vector that represents it. These vectors are the actual input for the Encoder.
3. **Encoder** several neural layers that model the token-level interactions of the input sequence to enhance meaning representation. The output of the encoder is a set of **H**idden layers, the vector representation of the ingested sequence.
5. **Output Layer:** the final encoder layer (which we depict as a sequence **H**'s in the figure) contains arguably the best token-level representations that encode syntactic and semantic properties of each token, but this time each vector is already contextualized with the specific sequence.
6. *OPTIONAL* **Classifier Layer:** an additional classifier can be connected on top of the BERT token vectors which are used as features for performing a downstream task. This can be used to classify at the text level, for example sentiment analysis of a sentence, or at the token-level, for example Named Entity Recognition.

![BERT Architecture](fig/bert3.png)

BERT uses (self-) attention, which is very useful to capture longer-range word dependencies such as correference, where, for example, a pronoun can be linked to the noun it refers to previously in the same sentence. See the following example:

![The Encoder Self-Attention Mechanism](fig/trans5.png)


## BERT for Word-Based Analysis

Let's see how these components can be manipulated with code. For this we will be using the HugingFace's _transformers_ python library.
The first two main components we need to initialize are the model and tokenizer. The HuggingFace hub contains thousands of models based on a Transformer architecture for dozens of tasks, data domains and also hundreds of languages. Here we will explore the vanilla English BERT which was how everything started. We can initialize this model with the next lines:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")
```

### BERT Tokenizer

We start with a string of text as written in any blog, book, newspaper etcetera. The `tokenizer` object is responsible of splitting the string into recognizable tokens for the model and embedding the tokens into their vector representations

```python
text = "Maria loves Groningen"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
```

The print shows the `encoded_input` object returned by the tokenizer, with its attributes and values. The `input_ids` are the most important output for now, as these are the token IDs recognized by BERT

```
{
    'input_ids': tensor([[  101,  3406,  7871,   144,  3484, 15016,   102]]), 
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]]), 
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])
}

```

 NOTE: the printing function shows transformers objects as dictionaries; however, to access the attributes, you must use the python object syntax, such as in the following example:

```python
print(encoded_input.input_ids.shape)
```
Output:

`torch.Size([1, 7])`

The output is a 2-dimensional tensor where the first dimention contains 1 element (this dimension represents the batch size), and the second dimension contains 7 elements which are equivalent to the 7 tokens that BERT generated with our string input.

In order to see what these Token IDs represent, we can _translate_ them into human readable strings. This includes converting the tensors into numpy arrays and converting each ID into its string representation:

```python
token_ids = list(encoded_input.input_ids[0].detach().numpy())
string_tokens = tokenizer.convert_ids_to_tokens(token_ids)
print("IDs:", token_ids)
print("TOKENS:", string_tokens)
```

`IDs: [101, 3406, 7871, 144, 3484, 15016, 102]`

`TOKENS: ['[CLS]', 'Maria', 'loves', 'G', '##ron', '##ingen', '[SEP]']`


::: callout

In the case of wanting to obtain a single vector for *enchanting*, you can average the three vectors that belong to the token pieces that ultimately form that word. For example:

```python
import numpy as np
tok_en = output.last_hidden_state[0][15].detach().numpy()
tok_chan = output.last_hidden_state[0][16].detach().numpy()
tok_ting = output.last_hidden_state[0][17].detach().numpy()

tok_enchanting = np.mean([tok_en, tok_chan, tok_ting], axis=0)
tok_enchanting.shape
```
We use the functions `detach().numpy()` to bring the values from the Pytorch execution environment (for example a GPU) into the main python thread and treat it as a numpy vector for convenvience. Then, since we are dealing with three numpy vectors we can average the three of them and end op with a single `enchanting` vector of 768-dimensions representing the average of `'en', '##chan', '##ting'`.

:::

### Polysemy in BERT
We can encode two sentences containing the word *note* to see how BERT actually handles polysemy (*note* means something very different in each sentence) thanks to the representation of each word now being contextualized instead of isolated as was the case with word2vec.


```python
# Search for the index of 'note' and obtain its vector from the sequence
note_index_1 = string_tokens.index("note")
note_vector_1 = output.last_hidden_state[0][note_index_1].detach().numpy()
note_token_id_1 = token_ids[note_index_1]

print(note_index_1, note_token_id_1, string_tokens)
print(note_vector_1[:5])
```
We are basically printing the tokenized sentence from the previous example and showing the index of the token `note` in the list of tokens. We are also printing the tokenID assigned to this token and the list of tokens. Finally, the last print shows the first five dimensions of the vector representing the token `note`.
```
12 3805 ['[CLS]', 'Maria', "'", 's', 'passion', 'for', 'music', 'is', 'clearly', 'heard', 'in', 'every', 'note', 'and', 'every', 'en', '##chan', '##ting', 'melody', '.', '[SEP]']
[0.15780845 0.38866335 0.41498923 0.03389652 0.40278202]
```

Let's encode now another sentence, also containing the word `note`, and confirm that the same token string, with the same assigned tokenID holds a vector with different weights:

```python
# Encode and then take the 'note' token from the second sentence
note_text_2 = "I could not buy milk in the supermarket because the bank note I wanted to use was fake."
encoded_note_2 = tokenizer(note_text_2, return_tensors="pt")
token_ids = list(encoded_note_2.input_ids[0].detach().numpy())
string_tokens_2 = tokenizer.convert_ids_to_tokens(token_ids)

note_index_2 = string_tokens_2.index("note")
note_vector_2 = model(**encoded_note_2).last_hidden_state[0][note_index_2].detach().numpy()
note_token_id_2 = token_ids[note_index_2]

print(note_index_2, note_token_id_2, string_tokens_2)
print(note_vector_2[:5])
```

```
12 3805 ['[CLS]', 'I', 'could', 'not', 'buy', 'milk', 'in', 'the', 'supermarket', 'because', 'the', 'bank', 'note', 'I', 'wanted', 'to', 'use', 'was', 'fake', '.', '[SEP]']
[ 0.5003222   0.653664    0.22919582 -0.32637975  0.52929205]
```

To be sure, we can compute the cosine similarity of the word *note* in the first sentence and the word *note* in the second sentence confirming that they are indeed two different representations, even when in both cases they have the same token-id and they are the 12th token of the sentence:

```python
from sklearn.metrics.pairwise import cosine_similarity

vector1 = np.array(note_vector_1).reshape(1, -1)
vector2 = np.array(note_vector_2).reshape(1, -1)

similarity = cosine_similarity(vector1, vector2)
print(f"Cosine Similarity 'note' vs 'note': {similarity[0][0]}")
```

With this small experiment, we have confirmed that the Encoder produces context-dependent word representations, as opposed to Word2Vec, where *note* would always have the same vector no matter where it appeared.

::: callout

When running examples in a BERT pre-trained model, it is advisable to wrap your code inside a `torch.no_grad():` context. This is linked to the fact that BERT is a Neural Network that has been trained (and can be further finetuned) with the Backpropagation algorithm. Essentially, this wrapper tells the model that we are not in training mode, and we are not interested in _updating_ the weights (as it would happen when training any neural network), because the weights are already optimal enough. By using this wrapper, we make the model more efficient as it does not need to calculate the gradients for an eventual backpropagation step, since we are only interested in what _comes out_ of the Encoder. So the previous code can be made more efficient like this:

```python
import torch 

with torch.no_grad():
    output = model(**encoded_input)
    print(output)
    print(output.last_hidden_state.shape)
```

:::


# BERT as a Language Model

As mentioned before, the main pre-training task of BERT is Language Modelling (LM): calculating the probability of a word based on the known neighboring words (yes, Word2Vec was also a kind of LM!). Obtaining training data for this task is very cheap, as all we need is millions of sentences from existing texts, without any labels. In this setting, BERT encodes a sequence of words, and predicts from a set of English tokens, what is the most likely token that could be inserted in the `[MASK]` position


![BERT Language Modeling](fig/bert1b.png)


We can therefore start using BERT as a predictor for word completion. From now own, we will learn how to use the `pipeline` object, this is very useful when we only want to use a pre-trained model for predictions (no need to fine-tune or do word-specific analysis). The `pipeline` will internally initialize both model and tokenizer for us and also merge back word pieces into complete words. 

In this case again we use `bert-base-cased`, which refers to the vanilla BERT English model. Once we declared a pipeline, we can feed it with sentences that contain one masked token at a time (beware that BERT can only predict one word at a time, since that was its training scheme). For example:


```python
from transformers import pipeline

def pretty_print_outputs(sentences, model_outputs):
    for i, model_out in enumerate(model_outputs):
        print("\n=====\t",sentences[i])
        for label_scores in model_out:
            print(label_scores)


nlp = pipeline(task="fill-mask", model="bert-base-cased", tokenizer="bert-base-cased")
sentences = ["Paris is the [MASK] of France", "I want to eat a cold [MASK] this afternoon", "Maria [MASK] Groningen"]
model_outputs = nlp(sentences, top_k=5)
pretty_print_outputs(sentences, model_outputs)
```

```
=====	 Paris is the [MASK] of France
{'score': 0.9807965755462646, 'token': 2364, 'token_str': 'capital', 'sequence': 'Paris is the capital of France'}
{'score': 0.004513159394264221, 'token': 6299, 'token_str': 'Capital', 'sequence': 'Paris is the Capital of France'}
{'score': 0.004281804896891117, 'token': 2057, 'token_str': 'center', 'sequence': 'Paris is the center of France'}
{'score': 0.002848200500011444, 'token': 2642, 'token_str': 'centre', 'sequence': 'Paris is the centre of France'}
{'score': 0.0022805952467024326, 'token': 1331, 'token_str': 'city', 'sequence': 'Paris is the city of France'}

=====	 I want to eat a cold [MASK] this afternoon
{'score': 0.19168031215667725, 'token': 13473, 'token_str': 'pizza', 'sequence': 'I want to eat a cold pizza this afternoon'}
{'score': 0.14800849556922913, 'token': 25138, 'token_str': 'turkey', 'sequence': 'I want to eat a cold turkey this afternoon'}
{'score': 0.14620967209339142, 'token': 14327, 'token_str': 'sandwich', 'sequence': 'I want to eat a cold sandwich this afternoon'}
{'score': 0.09997560828924179, 'token': 5953, 'token_str': 'lunch', 'sequence': 'I want to eat a cold lunch this afternoon'}
{'score': 0.06001955270767212, 'token': 4014, 'token_str': 'dinner', 'sequence': 'I want to eat a cold dinner this afternoon'}

=====	 Maria [MASK] Groningen
{'score': 0.24399833381175995, 'token': 117, 'token_str': ',', 'sequence': 'Maria, Groningen'}
{'score': 0.12300779670476913, 'token': 1104, 'token_str': 'of', 'sequence': 'Maria of Groningen'}
{'score': 0.11991506069898605, 'token': 1107, 'token_str': 'in', 'sequence': 'Maria in Groningen'}
{'score': 0.07722211629152298, 'token': 1306, 'token_str': '##m', 'sequence': 'Mariam Groningen'}
{'score': 0.0632941722869873, 'token': 118, 'token_str': '-', 'sequence': 'Maria - Groningen'}

```

When we call the `nlp` pipeline, requesting to return the `top_k` most likely suggestions to complete the provided sentences (in this case `k=5`). The pipeline returns a list of outputs as python dictionaries. Depending on the task, the fields of the dictionary will differ. In this case, the `fill-mask` task returns a score (between 0 and 1, the higher the score the more likely the token is), a tokenId, and its corresponding string, as well as the full "unmasked" sequence. 

In the list of outputs we can observe: the first example shows correctly that the missing token in the first sentence is _capital_, the second example is a bit more ambiguous, but the model at least uses the context to correctly predict a series of items that can be eaten (unfortunately, none of its suggestions sound very tasty); finally, the third example gives almost no useful context so the model plays it safe and only suggests prepositions or punctuation. This already shows some of the weaknesses of the approach.

We will next see the case of combining BERT with a classifier on top.


# BERT for Text Classification

The task of text classification is assigning a label to a whole sequence of tokens, for example a sentence. With the parameter `task="text_classification"` the `pipeline()` function will load the base model and automatically add a linear layer with a softmax on top. This layer can be fine-tuned with our own labeled data or we can also directly load the fully pre-trained text classification models that are already available in HuggingFace.


![BERT as an Emotion Classifier](fig/bert4.png)


Let's see the example of a ready pre-trained emotion classifier based on `RoBERTa` model. This model was fine-tuned in the Go emotions [dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions), taken from English Reddit and labeled for 28 different emotions at the sentence level. The fine-tuned model is called [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions). This model takes a sentence as input and ouputs a probability distribution over the 28 possible emotions that might be conveyed in the text. For example:

```python

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=3)

sentences = ["I am not having a great day", "This is a lovely and innocent sentence", "Maria loves Groningen"]
model_outputs = classifier(sentences)

pretty_print_outputs(sentences, model_outputs)
```

```
=====	 I am not having a great day
{'label': 'disappointment', 'score': 0.46669483184814453}
{'label': 'sadness', 'score': 0.39849498867988586}
{'label': 'annoyance', 'score': 0.06806594133377075}

=====	 This is a lovely and innocent sentence
{'label': 'admiration', 'score': 0.6457845568656921}
{'label': 'approval', 'score': 0.5112180113792419}
{'label': 'love', 'score': 0.09214121848344803}

=====	 Maria loves Groningen
{'label': 'love', 'score': 0.8922032117843628}
{'label': 'neutral', 'score': 0.10132959485054016}
{'label': 'approval', 'score': 0.02525361441075802}
```

This code outputs again a list of dictionaries with the `top-k` (`k=3`) emotions that each of the two sentences convey. In this case, the first sentence evokes (in order of likelihood) *dissapointment*, *sadness* and *annoyance*; whereas the second sentence evokes *love*, *neutral* and *approval*. Note however that the likelihood of each prediction decreases dramatically below the top choice, so perhaps this specific classifier is only useful for the top emotion.

::: callout

Finetunning BERT is very cheap, because we only need to train the _classifier_ layer,  a very small neural network, that can learn to choose between the classes (labels) for your custom classification problem, without needing a big amount of annotated data. This classifier is just a one-layer neural layer with a softmax that assigns a score that can be translated to the probability over a set of labels, given the input features provided by BERT, which _encodes_ the meaning of the entire sequence in its hidden states.

:::


![BERT as an Emotion Classifier](fig/bert4b.png)


# BERT for Token Classification

Just as we plugged in a trainable text classifier layer, we can add a token-level classifier that assigns a class to each of the tokens encoded by a transformer (as opposed to one label for the whole sequence). A specific example of this task is Named Entity Recognition, but you can basically define any task that requires to *highlight* sub-strings of text and classify them using this technique.

## Named Entity Recognition

Named Entity Recognition (NER) is the task of recognizing mentions of real-world entities inside a text. The concept of **Entity** includes proper names that unequivocally identify a unique individual (PER), place (LOC), organization (ORG), or other object/name (MISC). Depending on the domain, the concept can expanded to recognize other unique (and more conceptual) entities such as DATE, MONEY, WORK_OF_ART, DISEASE, PROTEIN_TYPE, etcetera... 

In terms of NLP, this boils down to classifying each token into a series of labels (`PER`, `LOC`, `ORG`, `O`[no-entity] ). Since a single entity can be expressed with multiple words (e.g. New York) the usual notation used for labeling the text is IOB (**I**nner **O**ut **B**eginnig of entity) notations which identifies the limits of each entity tokens. For example:


![BERT as an NER Classifier](fig/bert5.png)

This is a typical sequence classification problem where an imput sequence must be fully mapped into an output sequence of labels with global constraints (for example, there can't be an inner I-LOC label before a beginning B-LOC label). Since the labels of the tokens are context dependent, a language model with attention mechanism such as BERT is very beneficial for a task like NER.

Because this is one of the core tasks in NLP, there are dozens of pre-trained NER classifiers in HuggingFace that you can use right away. We use once again the `pipeline()` to run the model for predictions in your custom data, in this case with `task="ner"`. For example:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

ner_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang Schmid and I live in Berlin"

ner_results = ner_classifier(example)
for nr in ner_results:
    print(nr)
```

The code prints the following:

```
{'entity': 'B-PER', 'score': 0.9996068, 'index': 4, 'word': 'Wolfgang', 'start': 11, 'end': 19}
{'entity': 'I-PER', 'score': 0.999582, 'index': 5, 'word': 'Sc', 'start': 20, 'end': 22}
{'entity': 'I-PER', 'score': 0.9990482, 'index': 6, 'word': '##hm', 'start': 22, 'end': 24}
{'entity': 'I-PER', 'score': 0.9951691, 'index': 7, 'word': '##id', 'start': 24, 'end': 26}
{'entity': 'B-LOC', 'score': 0.99956733, 'index': 12, 'word': 'Berlin', 'start': 41, 'end': 47}
```

In this case the output of the pipeline is a list of dictionaries, each one representing only entity `IOB` labels at the BERT token level. IMPORTANT: this list is per wordPiece and NOT per *human word* even if the provided text is pre-tokenized. You can assume all of the tokens that don't appear in the output were labeled as no-entity, that is `"O"`. To recover the full-word entities you can initialize the pipeline with `aggregation_strategy="first"`:

```python
ner_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="first")
example = "My name is Wolfgang Schmid and I live in Berlin"

ner_results = ner_classifier(example)
for nr in ner_results:
    print(nr)
```
The code now prints the following:
```
{'entity_group': 'PER', 'score': 0.9995944, 'word': 'Wolfgang Schmid', 'start': 11, 'end': 26}
{'entity_group': 'LOC', 'score': 0.99956733, 'word': 'Berlin', 'start': 41, 'end': 47}
```


As you can see, entities aggregated at the Span Leven (instead of the Token Level). Word pieces are merged back into *human words* and also multiword entities are assigned a single entity label unifying the `IOB` labels into one. Depending on your use case you can request the pipeline to give different `aggregation_strateg[ies]`. More info about the pipeline can be found [here](https://huggingface.co/docs/transformers/main_classes/pipelines).

The next step is crucial: evaluate how does the pre-trained model actually performs in **your dataset**. This is important since the fine-tuned model could be overfitted to other custom benchmarks that do not share the characteristics of your dataset.

To observe this, we can first see the performance on the test portion of the dataset in which this classifier was trained, and then evaluate the same pre-trained classifier on a NER dataset form a different domain.


##  Model Evaluation

To perform evaluation in your data you can use again the `seqeval` package:

```python

from seqeval.metrics import classification_report
print(classification_report(gold_labels, model_predictions))

```

Since we took a classifier that was not trained for the book domain, the performance is quite poor. But this example shows us that classifiers performing very well on their own domain most of the times transfer poorly to other apparently similar datasets. 

The solution in this case is to use another of the great characteristics of BERT: fine-tuning for domain adaptation. It is possible to train your own classifier with relatively small data (given that a lot of linguistic knowledge was already provided during the language modeling pre-training). In the following section we will see how to train your own NER model and use it for predictions.
