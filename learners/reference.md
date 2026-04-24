---
title: 'Reference'
---

## Glossary

### Index of Terms

- [Accuracy](#accuracy)
- [Ambiguity](#ambiguity)
- [Attention Mechanism](#attention-mechanism)
- [Authorship Attribution](#authorship-attribution)
- [Backpropagation](#backpropagation)
- [Base Model](#base-model)
- [BERT](#bert)
- [Bias & Fairness](#bias-fairness)
- [Chunking](#chunking)
- [Co-reference Resolution](#coreference-resolution)
- [Compositionality](#compositionality)
- [Confusion Matrix](#confusion-matrix)
- [Context Window](#context-window)
- [Contextualized Word Representations](#contextualized-representations)
- [Continuous Bag-of-Words (CBOW)](#cbow)
- [Convolutional Neural Network (CNN)](#cnn)
- [Corpus / Corpora](#corpus)
- [Cosine Similarity](#cosine-similarity)
- [Data Formatting](#data-formatting)
- [Decoder](#decoder)
- [Deep Learning](#deep-learning)
- [Discreteness](#discreteness)
- [Distributional Semantics](#distributional-semantics)
- [Document Clustering](#document-clustering)
- [Domain-Specific Data](#domain-specific-data)
- [Downstream Task](#downstream-task)
- [ELMo](#elmo)
- [Embedder LLM vs. Generative LLM](#embedder-vs-generative)
- [Encoder](#encoder)
- [Entity Linking](#entity-linking)
- [F1-score](#f1-score)
- [FastText](#fasttext)
- [Fine-tuning](#fine-tuning)
- [Greedy Decoding](#greedy-decoding)
- [Guardrails](#guardrails)
- [Hallucination / Confabulation](#hallucination)
- [Hyperparameters](#hyperparameters)
- [Information Retrieval (IR)](#information-retrieval)
- [IOB Notation](#iob-notation)
- [Knowledge Base](#knowledge-base)
- [Language Identification](#language-identification)
- [Language Modeling](#language-modeling)
- [Large Language Model (LLM)](#llm)
- [Lemma / Lemmatization / Inflection](#lemmatization)
- [Long Short-Term Memory (LSTM)](#lstm)
- [Lowercasing](#lowercasing)
- [Machine Translation (MT)](#machine-translation)
- [Masked Language Model (MLM)](#mlm)
- [Morphological Ambiguity](#morphological-ambiguity)
- [Multi-layer Perceptron (MLP)](#mlp)
- [Multi-turn Conversation](#multi-turn)
- [Named Entity Recognition (NER)](#ner)
- [Natural Language](#natural-language)
- [Natural Language Processing (NLP)](#nlp)
- [Neural Network](#neural-network)
- [NLP Pipeline](#nlp-pipeline)
- [Non-determinism](#non-determinism)
- [Open-weights Model](#open-weights)
- [Overfitting](#overfitting)
- [Parameters](#parameters)
- [Paraphrasing](#paraphrasing)
- [Part-of-Speech (POS) Tagging](#pos-tagging)
- [Polysemy](#polysemy)
- [Post-training](#post-training)
- [Pragmatic Ambiguity](#pragmatic-ambiguity)
- [Pre-trained Model](#pretrained-model)
- [Precision](#precision)
- [Prompt Engineering](#prompt-engineering)
- [Question Answering (QA)](#question-answering)
- [Recall](#recall)
- [Recurrent Neural Network (RNN)](#rnn)
- [Reinforcement Learning from Human Feedback (RLHF)](#rlhf)
- [Relation Extraction](#relation-extraction)
- [RoBERTa](#roberta)
- [Self-Attention](#self-attention)
- [Semantic Ambiguity](#semantic-ambiguity)
- [Semantic Role Labeling (SRL)](#srl)
- [Sentence Segmentation](#sentence-segmentation)
- [Sentiment Analysis](#sentiment-analysis)
- [Skip-gram](#skip-gram)
- [Spam Filtering](#spam-filtering)
- [Sparsity](#sparsity)
- [Static Word Embedding](#static-word-embedding)
- [Subword Tokenization](#subword-tokenization)
- [Summarization](#summarization)
- [Supervised Fine-Tuning (SFT)](#sft)
- [Supervised Learning](#supervised-learning)
- [System / User / Assistant Roles](#conversation-roles)
- [Temperature](#temperature)
- [Text Classification](#text-classification)
- [Text Generation](#text-generation)
- [Token](#token)
- [Token Classification / Sequence Labeling](#token-classification)
- [Tokenization](#tokenization)
- [Top-K Sampling](#top-k-sampling)
- [Topic Modeling](#topic-modeling)
- [Training / Validation / Test Split](#train-val-test-split)
- [Training Cutoff](#training-cutoff)
- [Transfer Learning](#transfer-learning)
- [Transformer](#transformer)
- [True Positives / True Negatives / False Positives / False Negatives](#tp-tn-fp-fn)
- [ULMFiT](#ulmfit)
- [Unsupervised Learning](#unsupervised-learning)
- [Vector Space](#vector-space)
- [Word Embedding](#word-embedding)
- [Word Sense Disambiguation (WSD)](#wsd)
- [Word2Vec](#word2vec)

---

### NLP Fundamentals

<a id="natural-language"></a>**Natural Language**
: Human language as it is naturally spoken or written, as opposed to artificial languages such as programming languages. Natural languages are complex, [ambiguous](#ambiguity), and heavily context-dependent, which makes them challenging for computers to process.

<a id="nlp"></a>**Natural Language Processing (NLP)**
: An area of research and application focused on making human language processable by computers so that they can perform useful tasks. It spans a range of techniques, from simple word counts to complex deep learning architectures.

<a id="token"></a>**Token**
: A unit of text used as input by NLP algorithms. Typically a word, but potentially a subword piece, or character, depending on the task. The terms "token" and "word" are frequently used interchangeably, since modern tokenizers allow reconstructing the original text from its tokens.

<a id="tokenization"></a>**Tokenization**
: The task of segmenting a text into tokens. It is the first step to create structure from raw text. A good tokenizer handles edge cases such as abbreviations, contractions, dates, and URLs.

<a id="sentence-segmentation"></a>**Sentence Segmentation**
: The task of identifying where one sentence ends and the next begins. While trivial for human readers, it is a non-trivial NLP task because tokens like periods can mark abbreviations or decimal numbers rather than sentence boundaries.

<a id="corpus"></a>**Corpus / Corpora**
: A corpus (plural: corpora) is a collection of text documents used for training or evaluating NLP models.

<a id="knowledge-base"></a>**Knowledge Base**
: A structured repository of entities, concepts, and the relations between them, organized so that machines can query and reason over it. In NLP, knowledge bases such as [Wikidata](https://www.wikidata.org/) or [WordNet](https://wordnet.princeton.edu/) provide structured world knowledge that grounds tasks such as [Entity Linking](#entity-linking) and [Question Answering](#question-answering).

<a id="language-modeling"></a>**Language Modeling**
: The task of predicting the most likely next token given a sequence of preceding and/or subsequent tokens. It is the pre-training objective behind the latest NLP models, including Word2Vec, BERT and modern LLMs.

<a id="machine-translation"></a>**Machine Translation (MT)**
: The task of automatically translating text from one human language to another. It requires understanding the full meaning of the source sentence, since translation cannot be done word-by-word in isolation.

---

### Linguistic Properties of Text


<a id="compositionality"></a>**Compositionality**
: Language is built from layers: characters form words, words form phrases, and phrases form sentences, with meaning emerging at each level.

<a id="ambiguity"></a>**Ambiguity**
: The property of a word, phrase, or sentence admitting more than one interpretation. Ambiguity is pervasive in natural language and one of the central challenges for NLP systems, which must determine the intended meaning from context alone. It arises at multiple linguistic levels: [Morphological](#morphological-ambiguity), [Syntactic](#syntactic-ambiguity), [Semantic](#semantic-ambiguity), and [Pragmatic](#pragmatic-ambiguity).

<a id="morphological-ambiguity"></a>**Morphological Ambiguity**
: A form of [ambiguity](#ambiguity) occurring when the same word form can be parsed in two or more ways. For example, "unlockable" can mean either "capable of being unlocked" or "impossible to lock," depending on where the morpheme boundary falls (*un-lockable* vs. *unlock-able*). Disambiguating this happens almost automatically in human speakers, but poses a challenge when processing text with computational approaches.

<a id="syntactic-ambiguity"></a>**Syntactic Ambiguity**
: A form of [ambiguity](#ambiguity) occurring when a sentence can be parsed with more than one grammatical structure, each parse being a valid structure but entailing a different meaning.

<a id="semantic-ambiguity"></a>**Semantic Ambiguity**
: A form of [ambiguity](#ambiguity) occurring when syntactically identical sentences imply different meanings due to word sense (or contextual placement of an expression). "Drive the cat to the vet" uses "drive" very differently from "drive the car tomorrow", eventhough they have the same syntactic structure.

<a id="pragmatic-ambiguity"></a>**Pragmatic Ambiguity**
: A form of [ambiguity](#ambiguity) occurring when meaning depends on emphasis, tone, or context beyond the literal words. "I never said she stole my money" shifts meaning entirely depending on which word is stressed.

<a id="sparsity"></a>**Sparsity**
: The phenomenon where a word or concept of interest appears very rarely in a large body of text. Even with millions of words, particular terms may comprise only a tiny fraction of the corpus, making statistical estimation difficult.

<a id="discreteness"></a>**Discreteness**
: There is no inherent relationship between the written form of a word and its meaning. "Car" and "cat" differ by one letter yet refer to entirely unrelated things; "pizza" and "hamburger" share little written similarity yet are semantically close. This property motivates learning numeric vector representations of words rather than treating them as raw strings.

<a id="domain-specific-data"></a>**Domain-Specific Data**
: Text from a specialized field (e.g., medicine, law) in which vocabulary and word meanings differ from general language. The meaning of a concept can shift across domains: for example, "trial" in law refers to a court procedure, while in medicine it refers to a clinical experiment.

---

### NLP Tasks

<a id="text-classification"></a>**Text Classification**
: The task of assigning one or more predefined labels to a piece of text (a "document"). Documents can be sentences, paragraphs, or entire books depending on the task.

<a id="sentiment-analysis"></a>**Sentiment Analysis**
: A text classification task where the label captures the emotional tone of a text (e.g. positive, negative, or neutral).

<a id="language-identification"></a>**Language Identification**
: A text classification task that determines in which language a given input text is written.

<a id="spam-filtering"></a>**Spam Filtering**
: A text classification task that classifies messages (typically emails) as spam or legitimate based on their content.

<a id="authorship-attribution"></a>**Authorship Attribution**
: A text classification task that identifies the likely author of a text based on writing style and content, relying on the assumption that each author has a unique writing style.

<a id="token-classification"></a>**Token Classification / Sequence Labeling**
: The task of assigning a label to each individual token in a text. Because word meaning depends on surrounding context, models teng to tak einto account surrounding words to make a classification, hence the alternative names of Word-In-Context Classification or Sequence Classification.

<a id="pos-tagging"></a>**Part-of-Speech (POS) Tagging**
: A token classification task that assigns each word its grammatical role (e.g., noun, verb, adjective).

<a id="ner"></a>**Named Entity Recognition (NER)**
: A token classification task that identifies and categorizes real-world entities in text such as persons (PER), locations (LOC), organizations (ORG), and dates (DATE). For example, "Mary Shelley" is a person and "London" is a location.

<a id="wsd"></a>**Word Sense Disambiguation (WSD)**
: The task of determining which meaning of a word is intended in a given context. For example, "book" in "I read a book" (a publication) versus "I want to book a flight" (to reserve).

<a id="coreference-resolution"></a>**Co-reference Resolution**
: The task of determining which words or phrases in a text refer to the same real-world entity. For example, in "Mary is a doctor. She works at the hospital," the model must link "She" to "Mary."

<a id="chunking"></a>**Chunking**
: The task of segmenting a text into contiguous multi-word units ("chunks") that together represent a meaningful syntactic phrase.

<a id="relation-extraction"></a>**Relation Extraction**
: The task of identifying semantic relationships between named entities in a text. For example, from "Apple is based in California," the extracted relation is (Apple, based\_in, California).

<a id="entity-linking"></a>**Entity Linking**
: The task of disambiguating named entities in text by connecting them to their corresponding entries in a [knowledge base](#knowledge-base). For example, linking "Mary Shelley" to her Wikipedia page.

<a id="srl"></a>**Semantic Role Labeling (SRL)**
: The task of identifying "who did what to whom" in a sentence. For example, labeling agents, patients, and other participants of the event described by the text.

<a id="document-clustering"></a>**Document Clustering**
: An unsupervised task that groups similar documents together based on their content, without predefined labels.

<a id="topic-modeling"></a>**Topic Modeling**
: A specific form of document clustering that automatically discovers groups or "topics" for a collection of documents. For example by inferring clusters base on frequently co-occurring words across a collection.

<a id="information-retrieval"></a>**Information Retrieval (IR)**
: The task of finding relevant documents or passages from a large collection of unstructured text in response to a user query. Web search engines are the most prominent application.

<a id="text-generation"></a>**Text Generation**
: The task of producing new text from a given input, typically token by token conditioned on a premise and the previously generated output. It encompasses tasks such as [machine translation](#machine-translation), [summarization](#summarization), and [question answering](#question-answering). For example int machine translation the premise is a sentence in the source language which conditions how the sentence in the target language should start.

<a id="summarization"></a>**Summarization**
: A text generation task that produces a shorter version of a longer text. It can be extractive (selecting key sentences verbatim) or abstractive (generating new sentences that capture the main ideas).

<a id="question-answering"></a>**Question Answering (QA)**
: The task of generating or selecting an answer given a question and a context passage. It can be framed as classification (choosing from predefined options) or as generation (producing the answer from scratch).

<a id="paraphrasing"></a>**Paraphrasing**
: The task of generating a new sentence that conveys the same meaning as an original sentence but with different wording.

---

### Machine Learning & Deep Learning

<a id="supervised-learning"></a>**Supervised Learning**
: A machine learning approach where a model learns from labeled examples: input-output pairs curated by humans. The model learns to map new inputs to correct outputs and is evaluated on examples it has never seen before during training.

<a id="unsupervised-learning"></a>**Unsupervised Learning**
: A machine learning approach that finds structure in data without any human-provided labels. Common unsupervised NLP tasks include document clustering and topic modeling, where features are automatically extracted and data points that are close to each other belong to the same group(s).

<a id="deep-learning"></a>**Deep Learning**
: A subfield of machine learning that uses neural networks with many layers to learn complex patterns directly from data. Deep learning has driven the most significant NLP advances of the past decade.

<a id="neural-network"></a>**Neural Network**
: A pattern-finding machine made of interconnected layers of neurons, where each neuron holds a numerical value adjusted during training to maximize prediction accuracy. The deeper the network, the more complex the patterns it can learn.

<a id="backpropagation"></a>**Backpropagation**
: The algorithm used to train neural networks. It computes how much each weight contributed to the prediction error and adjusts weights to reduce that error, repeating until the model converges.

<a id="mlp"></a>**Multi-layer Perceptron (MLP)**
: The simplest deep neural network: a sequence of fully connected layers where every neuron in one layer connects to every neuron in the next.

<a id="rnn"></a>**Recurrent Neural Network (RNN)**
: A neural network designed for sequential data that processes tokens one at a time while maintaining a "memory" state of previously seen tokens. RNNs struggle with long-range dependencies in text.

<a id="lstm"></a>**Long Short-Term Memory (LSTM)**
: An improved RNN architecture that uses gating mechanisms to selectively retain or forget information across long sequences, addressing the vanishing gradient problem (completely 'forgetting' items from the beginning of a sequence) of standard RNNs.

<a id="cnn"></a>**Convolutional Neural Network (CNN)**
: Originally designed for image recognition, CNNs can be applied to text by sliding filters over sequences of word vectors to detect local patterns such as n-grams.

---

### Pre-trained Models & Transfer Learning

<a id="pretrained-model"></a>**Pre-trained Model**
: A model already trained on large amounts of data for a general task, which can be loaded and applied to new datasets without further training. Pre-trained models save significant time and compute by starting with broad linguistic knowledge rather than random weights.

<a id="fine-tuning"></a>**Fine-tuning**
: The process of continuing the training of a pre-trained model on a smaller, task-specific dataset. Fine-tuning adapts the general knowledge of the pre-trained model to the nuances of a specific domain or task, usually requiring much less data and compute than training from scratch.

<a id="transfer-learning"></a>**Transfer Learning**
: The practice of reusing knowledge from a model trained on one task to improve performance on a related [downstream task](#downstream-task). Fine-tuning is the most common form of transfer learning in NLP.

<a id="downstream-task"></a>**Downstream Task**
: The specific NLP task a pre-trained model is adapted to solve after pre-training — for example, [sentiment analysis](#sentiment-analysis), [named entity recognition](#ner), or [machine translation](#machine-translation). The term contrasts with the upstream pre-training objective (e.g., [language modeling](#language-modeling)) and reflects the typical [transfer learning](#transfer-learning) workflow: pre-train on general data, then fine-tune on the downstream task of interest.

<a id="distributional-semantics"></a>**Distributional Semantics**
: An approach to meaning based on the principle that "words that appear in similar contexts have similar meanings." By studying word co-occurrence patterns in large text collections, we can infer that "pizza" and "hamburger" are semantically closer than "car" and "cat", without any hand-crafted definitions.

---

### Text Preprocessing & NLP Pipelines

<a id="nlp-pipeline"></a>**NLP Pipeline**
: A sequence of operations applied to raw text to produce the output required by an NLP task. Components can include preprocessing steps, rule-based modules, machine learning models, and output formatters. Errors introduced at early stages propagate through the rest of the pipeline.

<a id="data-formatting"></a>**Data Formatting**
: The first preprocessing step: obtaining a clean, consistent text representation from diverse raw sources such as PDFs, Word documents, or web pages. Common operations include removing irrelevant characters, replacing line breaks, and stripping HTML tags.

<a id="lowercasing"></a>**Lowercasing**
: Converting all text to lowercase so that "Dog" and "dog" are treated as the same word. Useful for count-based methods and embedding training, but can hurt tasks like Named Entity Recognition that rely on capitalization as a signal (e.g., "Apple" the company vs. "apple" the fruit).

<a id="lemmatization"></a>**Lemma / Lemmatization / Inflection**
: An *inflection* is a surface variation of a root word. For example, "eating," "ate," and "eaten" are all inflections of "eat." The *lemma* is the canonical dictionary form of a word. *Lemmatization* is the preprocessing step that maps each token to its lemma, useful for count-based methods that benefit from collapsing different surface forms of the same concept.

---

### Word Representations

<a id="word-embedding"></a>**Word Embedding**
: A dense numeric vector that represents a word in a continuous vector space. Words used in similar contexts receive similar vectors, encoding semantic relationships as geometric distances.

<a id="word2vec"></a>**Word2Vec**
: A shallow neural network model that learns word embeddings by training on a language modeling objective, introduced by Mikolov et al. (2013). It represents each word as a fixed-size vector; similar words end up close together in the resulting vector space.

<a id="cbow"></a>**Continuous Bag-of-Words (CBOW)**
: One of the two Word2Vec training algorithms. CBOW predicts a target word from its surrounding context words.

<a id="skip-gram"></a>**Skip-gram**
: The second Word2Vec training algorithm. Skip-gram does the reverse of CBOW: given a target word, it predicts the surrounding context words. It tends to perform better on less frequent words.

<a id="vector-space"></a>**Vector Space**
: The multi-dimensional space in which word embeddings are positioned. Each word is a point in this space, and geometric proximity or angle between points reflects semantic similarity.

<a id="cosine-similarity"></a>**Cosine Similarity**
: A measure of similarity between two vectors, equal to the cosine of the angle between them. It ranges from −1 (opposite directions) to 1 (identical directions) and is the standard metric for comparing word embeddings because it is independent of vector length.

<a id="static-word-embedding"></a>**Static Word Embedding**
: A word embedding that assigns a single fixed vector to each word regardless of context. Word2Vec is a static embedding: "bank" always has the same vector whether it appears in a financial or geographical context. This is the key limitation that contextualized models like BERT address.

<a id="polysemy"></a>**Polysemy**
: The property of a word having multiple distinct meanings depending on context. For example, "note" means something very different in "please note that" and "this bank note is fake." Static embeddings cannot distinguish these meanings; contextualized models like BERT produce different vectors for each usage.

<a id="contextualized-representations"></a>**Contextualized Word Representations**
: Word vectors that change depending on the sentence in which a word appears. Unlike static embeddings, a contextualized model assigns a different vector to "bank" in "river bank" and "bank account," capturing the actual intended meaning in each case.

---

### Transformer Architecture

<a id="transformer"></a>**Transformer**
: A deep neural network architecture introduced in "Attention Is All You Need" (Vaswani et al., 2017), originally designed for [machine translation](#machine-translation). It uses attention mechanisms rather than recurrence, enabling highly parallelizable training at scale. It is the foundational architecture behind BERT, GPT, and most modern LLMs.

<a id="encoder"></a>**Encoder**
: The Transformer component that reads an input sequence and produces rich, context-aware representations of it. Encoder-only models like BERT are specialized for understanding and representing text.

<a id="decoder"></a>**Decoder**
: The Transformer component that generates output tokens one by one, conditioned on the encoder's representations and the tokens already produced. Decoder-only models like GPT and LLaMA are used for text generation.

<a id="attention-mechanism"></a>**Attention Mechanism**
: A neural network component that computes a relevance score between each pair of tokens in a sequence, allowing the model to focus on the most informative tokens for a given prediction. It enables transformers to capture long-range dependencies that RNNs struggle with.

<a id="self-attention"></a>**Self-Attention**
: A form of attention where each token attends to all other tokens in the same sequence  (including itself) to build a context-aware representation. Self-attention is the core building block of the Transformer Encoder.

<a id="bert"></a>**BERT (Bidirectional Encoder Representations from Transformers)**
: A pre-trained Transformer Encoder model introduced by Google in 2018. It generates contextualized word representations by attending to both left and right context simultaneously.

<a id="mlm"></a>**Masked Language Model (MLM)**
: A pre-training task in which some tokens in an input sentence are replaced with a `[MASK]` placeholder and the model learns to predict the original token from the surrounding context. It is one of BERT's two pre-training objectives.

<a id="subword-tokenization"></a>**Subword Tokenization**
: A tokenization strategy that splits words into smaller statistically-motivated pieces. For example, BERT splits "Groningen" into `['G', '##ron', '##ingen']`. It allows the model to handle rare or previously unseen words by combining familiar subword pieces, unlike word-level tokenizers that fail on unknown words.

<a id="iob-notation"></a>**IOB Notation**
: A labeling scheme used in token classification tasks such as NER to mark entity boundaries. `B-` (Beginning) marks the first token of an entity, `I-` (Inside) marks subsequent tokens of the same entity, and `O` (Outside) marks non-entity tokens.

<a id="fasttext"></a>**FastText**
: A word embedding model by Facebook (Joulin et al., 2016) that extends Word2Vec by representing words as bags of character n-grams. This allows it to generate embeddings for unseen words and to handle misspellings or morphologically rich languages.

<a id="elmo"></a>**ELMo (Embeddings from Language Models)**
: A contextualized word representation model (Peters et al., 2018) that produces different vectors for the same word depending on its context, using bidirectional LSTM networks. ELMo was a key precursor to BERT, introducing effective transfer learning to NLP.

<a id="ulmfit"></a>**ULMFiT (Universal Language Model Fine-Tuning)**
: A transfer learning method (Howard & Ruder, 2018) showing that a single language model pre-trained on general text can be fine-tuned for many [downstream NLP tasks](#downstream-task). It established the pre-train-then-fine-tune paradigm that BERT would later scale dramatically.

<a id="roberta"></a>**RoBERTa**
: A BERT variant (Liu et al., 2019) trained with improved procedures: more data, longer training, and removal of the Next Sentence Prediction objective. RoBERTa consistently outperforms the original BERT on many benchmarks.

---

### Model Evaluation

<a id="train-val-test-split"></a>**Training / Validation / Test Split**
: The practice of dividing a labeled dataset into three non-overlapping subsets. The *training set* teaches the model; the *validation set* guides hyperparameter tuning; the *test set* provides the final, unbiased performance estimate. Without this separation, you risk measuring how well the model memorized training data rather than how well it generalizes.

<a id="overfitting"></a>**Overfitting**
: When a model memorizes training data rather than learning generalizable patterns. An overfitted model performs well on training data but poorly on new, unseen examples.

<a id="accuracy"></a>**Accuracy**
: The proportion of all predictions that were correct (TP + TN) divided by the total number of instances. Accuracy can be misleading when class sizes are imbalanced.

<a id="precision"></a>**Precision**
: Of all the predictions made for a given class, what fraction were actually correct? Precision = TP / (TP + FP). High precision means few false alarms.

<a id="recall"></a>**Recall**
: Of all the actual instances of a given class in the data, what fraction did the model correctly identify? Recall = TP / (TP + FN). High recall means few missed detections.

<a id="f1-score"></a>**F1-score**
: The harmonic mean of precision and recall, providing a single balanced metric when the two trade off against each other.

<a id="tp-tn-fp-fn"></a>**True Positives (TP) / True Negatives (TN) / False Positives (FP) / False Negatives (FN)**
: The four possible outcomes when comparing a model's prediction to the ground truth for a given class. TP and TN are correct predictions; FP ("false alarm") and FN ("miss") are errors.

<a id="confusion-matrix"></a>**Confusion Matrix**
: A table that compares predicted labels against true labels across all classes. Perfect predictions appear on the diagonal; off-diagonal cells reveal systematic errors, confusion between specific classes, and class imbalance.

<a id="hyperparameters"></a>**Hyperparameters**
: Settings that control how a model is trained or how it generates output. For example, the number of training epochs, the learning rate, or the temperature during text generation. Hyperparameters are chosen before training and are not learned from data.

---

### Large Language Models

<a id="llm"></a>**Large Language Model (LLM)**
: A generative transformer-based language model trained at massive scale (hundreds of billions of parameters on vast amounts of text) and post-trained to interact with users conversationally. LLMs can address a wide range of NLP tasks without task-specific training.

<a id="parameters"></a>**Parameters**
: The numerical weights inside a neural network learned during training (numbers inside matrices that get mathematically manipulated). The "large" in LLM refers primarily to the number of parameters: modern LLMs have hundreds of billions, compared to BERT's ~110 million.

<a id="context-window"></a>**Context Window**
: The maximum number of tokens an LLM can process in a single interaction. BERT was limited to 512 tokens; some current LLMs can handle several million. A larger context window allows the model to reason over longer documents and conversation histories.

<a id="base-model"></a>**Base Model**
: A language model trained on raw text data with a next-token prediction objective, before any post-training. A base model is not yet suitable for direct conversation, it is merely the starting point that provides linguistic knowledge for further fine-tuning in more specific [downstream tasks](#downstream-task).

<a id="post-training"></a>**Post-training**
: The set of fine-tuning steps applied after base model training to make an LLM useful as a conversational assistant. It includes Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and safety training.

<a id="sft"></a>**Supervised Fine-Tuning (SFT)**
: A post-training step where the model is trained on curated input-output pairs that demonstrate desired behavior, such as instruction-following or question-answering exchanges.

<a id="rlhf"></a>**Reinforcement Learning from Human Feedback (RLHF)**
: A post-training technique where human raters compare model outputs and the model is updated to produce responses that humans prefer. RLHF is central to making LLMs sound helpful and to reducing harmful outputs.

<a id="open-weights"></a>**Open-weights Model**
: A model whose trained parameters are publicly available for download and local use, even if the training data or architecture code are not fully disclosed. Open-weights models (e.g., LLaMA) can be run and fine-tuned locally, unlike proprietary API-only models.

<a id="embedder-vs-generative"></a>**Embedder LLM vs. Generative LLM**
: *Embedder LLMs* (encoder-only models such as BERT) are optimized for producing meaningful text representations used in tasks like similarity search. *Generative LLMs* (decoder-only models such as GPT or LLaMA) produce new text token by token. Most modern chat assistants are generative LLMs.

---

### Prompting & Generation Control

<a id="prompt-engineering"></a>**Prompt Engineering**
: The practice of carefully designing the input text given to an LLM to obtain the desired output. It includes choosing the right instructions, providing examples, and structuring system, user, and assistant roles.

<a id="conversation-roles"></a>**System / User / Assistant Roles**
: The three core roles in a structured LLM conversation. The *system* message sets the model's general behavior. *User* messages contain human input or questions. *Assistant* messages are the model's generated responses. These roles are internally converted into special tokens and processed together as a single sequence.

<a id="temperature"></a>**Temperature**
: A hyperparameter controlling the randomness of token selection during generation. At 0.0, the model always picks the most likely next token (deterministic); higher values produce more varied but potentially less coherent responses.

<a id="top-k-sampling"></a>**Top-K Sampling**
: A generation strategy that at each step restricts the model's choices to the K most probable next tokens and samples from that reduced set, balancing diversity and coherence.

<a id="greedy-decoding"></a>**Greedy Decoding**
: A generation strategy that always selects the single most probable next token at each step. It is deterministic and fast but often produces repetitive text.

<a id="multi-turn"></a>**Multi-turn Conversation**
: An interaction with an LLM consisting of multiple back-and-forth exchanges, where the full conversation history is passed as context at each turn.

---

### LLM Behavior & Limitations

<a id="hallucination"></a>**Hallucination / Confabulation**
: The generation of content that is factually incorrect, fabricated, or not grounded in any source, despite sounding fluent and confident. It is a fundamental limitation of generative models and the primary reason LLM outputs must always be verified.

<a id="non-determinism"></a>**Non-determinism**
: The property that the same prompt can produce different outputs across runs due to random sampling during generation. Even setting temperature to 0.0 may not guarantee identical outputs across different hardware or software configurations.

<a id="bias-fairness"></a>**Bias & Fairness**
: LLMs inherit biases present in their pre-training data and post-training labels. For example, they may systematically associate certain professions with specific genders. These biases can propagate to downstream applications, making evaluation on diverse data essential.

<a id="training-cutoff"></a>**Training Cutoff**
: The date beyond which an LLM has no knowledge, because its training data was collected up to that point. The model may answer questions about events after the cutoff incorrectly and often without any warning.

<a id="guardrails"></a>**Guardrails**
: Constraints added during post-training to prevent a model from generating harmful, offensive, or clearly false outputs. Guardrails are imperfect and could introduce further unintended biases.

---

## External References

### Key Papers

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space* (Word2Vec). <https://arxiv.org/pdf/1301.3781>
- Vaswani et al. (2017). *Attention Is All You Need* (Transformer architecture). <https://arxiv.org/pdf/1706.03762>
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. <https://aclanthology.org/N19-1423.pdf>
- Joulin et al. (2016). *Bag of Tricks for Efficient Text Classification* (FastText). <https://arxiv.org/pdf/1607.01759>
- Peters et al. (2018). *Deep Contextualized Word Representations* (ELMo). <https://aclanthology.org/N18-1202.pdf>
- Howard & Ruder (2018). *Universal Language Model Fine-tuning for Text Classification* (ULMFiT). <https://aclanthology.org/P18-1031.pdf>
- Lenci (2018). *Distributional Models of Word Meaning* (survey on distributional semantics). <https://arxiv.org/pdf/1905.01896>
- Huang et al. (2024). *Survey on Hallucination in Large Language Models* (confabulation). <https://arxiv.org/abs/2406.04175>

### Tools & Libraries

- [spaCy](https://github.com/explosion/spaCy) — Industrial-strength NLP library for Python with pre-trained models for tokenization, POS tagging, NER, and more.
- [NLTK](https://github.com/nltk/nltk) — The Natural Language Toolkit; a comprehensive Python platform for working with human language data.
- [Gensim](https://radimrehurek.com/gensim/) — Python library for topic modeling and word embedding training, including Word2Vec.
- [Stanza](https://github.com/stanfordnlp/stanza) — Stanford NLP Group's Python library supporting 70+ languages with tokenization, POS tagging, NER, and dependency parsing.
- [Flair](https://github.com/flairNLP/flair) — A simple framework for state-of-the-art NLP, including NER, POS tagging, and text classification.
- [FastText](https://github.com/facebookresearch/fastText) — Facebook's library for efficient text classification and subword-aware word representations.
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — The standard Python library for loading and using pre-trained transformer models (BERT, RoBERTa, GPT, and more).
- [Ollama](https://github.com/ollama/ollama) — An open-source tool for downloading and running LLMs locally on your own machine.
- [LangChain](https://docs.langchain.com/) — A Python framework for building applications with LLMs, providing a unified interface across many models and providers.
- [scikit-learn](https://scikit-learn.org/) — A widely used Python library for machine learning, including evaluation metrics such as `classification_report`.
- [seqeval](https://github.com/chakki-works/seqeval) — A Python library for evaluating sequence labeling tasks such as NER.
- [Tiktokenizer](https://tiktokenizer.vercel.app/) — A web app for visualizing how different tokenizers split text into tokens.

### Datasets & Linguistic Resources

- [HuggingFace Hub](https://huggingface.co/) — A central platform for sharing pre-trained models and datasets for NLP and machine learning.
- [HuggingFace Datasets](https://huggingface.co/datasets) — A large collection of NLP datasets covering text classification, question answering, language modeling, and more.
- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) — A dataset of English Reddit comments annotated for 28 emotion categories, used in the emotion classification examples in Episode 3.
- [WordNet](https://wordnet.princeton.edu/) — A large English lexical database where words are grouped into synonym sets (synsets) linked by semantic relations.
- [Universal Dependencies](https://universaldependencies.org/) — Syntactically annotated treebanks for 100+ languages with a consistent scheme for morphological and syntactic properties.
- [PropBank](https://propbank.github.io/) — A corpus annotated with semantic propositions (who did what to whom), useful for Semantic Role Labeling.
- [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/) — A lexical resource providing information about the semantic frames underlying word meanings, including their roles and relations.
- [BabelNet](https://babelnet.org/) — A multilingual lexical resource combining WordNet and Wikipedia, covering concepts and relations in many languages.
- [Wikidata](https://www.wikidata.org/) — A free, open knowledge base with structured data about entities, their properties, and relationships; used to enrich NLP applications.
- [Dolma](https://github.com/allenai/dolma) — An open dataset of 3 trillion tokens from diverse sources (web, books, code, encyclopedic material) used to train English LLMs.

### Further Reading

- Ruder, S. (2020). *NLP Beyond English* — A blog post surveying challenges and opportunities for NLP in non-English and minority languages. <https://www.ruder.io/nlp-beyond-english/>
- McCormick, C. (2016). *Word2Vec Tutorial: The Skip-Gram Model* — An intuitive walkthrough of how Word2Vec is trained. <https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/>
- HuggingFace. *Fine-tuning a pre-trained model* — Official tutorial for adapting BERT-like models to custom classification tasks. <https://huggingface.co/docs/transformers/v4.57.1/en/training#fine-tuning>
- spaCy. *Models & Languages* — Overview of available pre-trained spaCy models across many languages. <https://spacy.io/models>
- HuggingFace. *Text Generation Strategies* — Documentation on decoding strategies (greedy, sampling, beam search) in the Transformers library. <https://huggingface.co/docs/transformers/generation_strategies>
