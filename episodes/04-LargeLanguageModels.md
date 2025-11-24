---
title: "Using large language models"
teaching: 60
exercises: 60
---

## Background

Chat assistants like [ChatGPT](https://chatgpt.com/), [Gemini](https://gemini.google.com/) and [Claude](http://claude.ai) are products widely used today for tasks such as content generation, question answering, research and software development among many others. These products, also known as Large Language Modesl (LLMs), are based on the same Transformer architecture, with several enhancements such as feeding considerably large datasets, combining several models to generate an output, and several other post-training techniques that enhance the predictions to fit the product expectations. It is undeniable that the rapid rise of such models has had quite a disruptive and strong impact. But what are these models exactly? How do they work 'under the hood'? And how can one use them programmatically, in a responsible and effective way?

<img src="fig/llm-logos/anthropic.png" alt="Company A" width="80"/> <img src="fig/llm-logos/alibaba.png" alt="Company B" width="80"/> <img src="fig/llm-logos/xai.jpg" alt="Company C" width="80"/> <img src="fig/llm-logos/zhipu.png" alt="Company C" width="150"/> <img src="fig/llm-logos/google.png" alt="Company C" width="80"/> <img src="fig/llm-logos/openai.jpg" alt="Company C" width="150"/>

<img src="fig/llm-logos/nvidia.png" alt="Company D" width="80"/> <img src="fig/llm-logos/deepseek.png" alt="Company E" width="80"/> <img src="fig/llm-logos/huggingface.png" alt="Company F" width="80"/> <img src="fig/llm-logos/meta.png" alt="Company C" width="150"/>

Now that we know that dealing with written language means extracting structure from unstructured text (NLP Task), we can understand there is a rationale behind the kind of tasks LLMs claim to tackle. Everytime we prompt a chat model, we are assigning them a task that needs to get solved. The question that interests us is not about LLMs being "intelligent" or not (eventhough that could be a very interesting topic on its own, is not the goal of this course); we should rather focus on using LLMs as yet a new *automatic tool*, and like with any other Machine Learning or NLP tool, we are still dealing here with a model that takes an input and delivers an output for that input. The only thing that changed is the complexity of the input-output pairs; hence, the complexity for validating the outputs increased accordingly. 

Our duty as NLP practitioners remains, and we must keep asking the same questions: 
- How can I shape the task so I obtain the information I need? 
- How do I measure what proportion of the outputs are right? 
- How do I know if this will behave as expected with unexpected inputs? 
- **In short: How do I evaluate?**  

This episode is a gentle introduction to LLMs which aims to equip you with knowledge of the underpinnings of LLMs based on transformers architecture, as well as practical skills to start programmatically working with LLMs in your own projects, without necessarily relying on proprietary products and platforms.


## What are Large Language Models (LLMs)?
Large language models (LLMs) are transformer-based language models that are specialised to interpret and generate text, and to converse in a conversational-like manner with humans. The text that they generate are mostly natural language but can, in theory, constitute any character or symbol sequence such as software code. The term *Large* was appended to the well known *Language Model* term to highlight the scale on which this architectures are trained. To give an example, BERT in 2018 was considered a big model and had roughly 100 million parameters; GPT-2 in 2019 had 1.5 billion parameters, and GPT-3 was published in 2020 as a model with 175 billion parameters, and so forth. 

::: callout
Since we already learned about the vanilla transformer architecture, in this episode we will focus on the most recent language models, and we will keep calling them LLMs, eventhough they are not necessarily that large anymore. 

Given the hype around the field, people keep calling any new model an LLM, and we will keep using the term to avoid confusion. Have in mind, however, that especially recent models are being published with the particular goal of reducing parameter size while retaining the performance of the larger models, some of them are even less than 1 billion parameters already! This is good news for the open source engineers and researchers, because such advancements mean we can now make use of the new capabilities of language models in our own local servers, and even our laptops, without needing to pay fees or compromise the privacy of our data and experiments.
:::

### Examples of LLMs 

Many different LLMs have been, and continue to be, developed. There are both proprietary and open-source varieties. Real open-source varieties should make the whole path of model creation available: inform exactly which data was used to train the model, including filters for data quality; give a detailed explanation of the architecture and hypermarameters, including the code used to train them, and of course make them free, open and accessible online. Unfortunately completely ope source models are scarce, but the partially open source number models keeps growing. Below is a summary of some relevant LLMs together with their creators, chat assistant interfaces, and proprietary status:

<img src="fig/llm_table4.png" alt="LLMs table" width="1000" />


### Interacting with an LLM

Before exploring how we can invoke LLMs programmatically to solve the kinds of tasks abve, let us setup and load our first LLM. We will keep using the `transformers` library, just as with the BERT examples.

#### Import Libraries
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
```

#### Load and setup the model (LLM)

Let's load a open source lightweight `SmolLM2-135M-Instruct`, as you might have guessed it is a model with 135M parameters which has been finetuned for following instructions, therefore you can use it as a chat assistant:

```python
# We'll use SmolLM-135M - an open, small, fast model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct" # fine-tuned assistant model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if model is loaded correctly
print(f"Model loaded! It has {model.num_parameters():,} parameters")
```

#### Basic Text Generation

Let's perform inference with the LLM to generate some text.

```python
# Set pad_token_id to eos_token_id to avoid warnings
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.eos_token_id
    
# We Build pipeline as usual, this time the task is `text_generation` (a decoder model)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "Where is Groningen located?"

# We can first look at how the text is also being converted into a sequence of tokens before generating
print(tokenizer(prompt))
print(tokenizer.convert_ids_to_tokens(tokenizer(prompt)["input_ids"]))
```
This gives the following output:
```
{'input_ids': [9576, 314, 452, 992, 45670, 3807, 47], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
['Where', 'Ġis', 'ĠG', 'ron', 'ingen', 'Ġlocated', '?']
```

Now we ask the pipeline to generate the tokens to complete the given input. Remember that the pipeline post-processes the output token for us and shows us the text string under `generated_text` field:
```python
response = llm(prompt)
print(response)
```

This is not a fully deterministic ouput, so the content of the response will vary, but the structure is always the same: a python list, containing as first element a dictionary. The keys of the dictionary can vary depending on the parameters given to the pipeline, but it will always have at least the key `generated_text` which gives you the models ouptut in the form of a python string:

```
[{'generated_text': 'Where is Groningen located?\nGroningen is a municipality in the province of Drenthe in the North Holland region of the Netherlands. As of the 2011 census, the municipality had a population of 122,683.\nWhat is the name of the river that connects Groningen to Utrecht?'}]
```

We can obtain only the text if we extract it directly:

```python
text = response[0]["generated_text"]
print(text)
```

Note that the model "repeats" the question before completing it with the answer. This happens because in reality the LLM is a Transformer Decoder, which tokenizes your prompt, processes it token by token, and then keeps *generating new tokens*. Because the model was trained to act as an assistant in a conversation, the most likely tokens that follow a questions are an answer to that question. But it is possible that the model does not reply at all.

![Generative LLMs correspond to the Decoder component of the Transformer architecture](fig/llms_vs_bert2.png)

#### Step 3: Multi-turn Conversation Generation

When dealing with these kind of *assistant LLMs* it is a better practice to look at the input as a conversation. A conversation can be defined as a list of *messages* that the pipeline knows how to internally tokenize and generate a better answer given the context provided:

```python
messages = [
  {"role": "system", "content": "You are a helpful assistant. Give short straight answers" },
  {"role": "user", "content": "Where is Groningen located?" },
]
response = llm(messages, return_full_text=False)[0]["generated_text"]
print(response)
```

This method makes it more likely to get an answer everytime the model is executed. Remember this is not completely deterministic, so you can run it several times. (Spoiler: Expect several falsehoods to be stated in the generations). You should get something like:

```
Groningen is located in the northwest corner of the Netherlands.
```

We set the `return_full_text=False` flag to False to get the response as a string. This parameter is by default set to True and in that case it will return the whole conversation so you can keep using the history for posterior interactions if you want to:

```python
messages = [
  {"role": "system", "content": "You are a helpful assistant. Give short straight answers" },
  {"role": "user", "content": "Where is Groningen located?" },
]
response = llm(messages)[0]["generated_text"]
print(response)
```
This gives a multi-turn conversation as a list of string messages, such as:

```python
[{
    'role': 'system', 
    'content': 'You are a helpful assistant. Give short straight answers'
 }, 
 {
    'role': 'user', 
    'content': 'Where is Groningen located?'
 }, 
 {
    'role': 'assistant', 
    'content': 'Groningen is located in the northern part of the Netherlands.'
}]

```

::: callout
You may have noticed there are three main `roles` that most LLMs recognize in an interaction:

- **System:** This role is normally only used once at the beginning of an interaction, it contains the general prompting strategies (instructions) that your model should follow. Think of it as the user manual (where you explain what the task is) that the LLM will read before performing the actual task. This is where *prompt engineering* happens.
- **User:** also referred sometimes as **Human**, includes all messages *human generated* in the dialogue are given the user role. In a multi-turn dialogue, each utterance you submit will be given the user role, to distinguish them from the automatically generated sequences. These are quesitons submitted ot the model, or *input instances* for which you expect an output.
- **Assistant:**  All messages *automatically generated* based on the model text generation capabilities, these are the *model outputs*.

Different models might recognize also other roles, but these are the core ones. You can always pass directly a text prompt, even if you do only one-turn interactions with the LLMs (which many times is the recommended option to avoid biasing model repsonses based on previous ones), but it a good practice to follow these roles since the start, to make your code more scalable. These roles are internally translated into special tokens and processed together as a single sequence. See a [tiktokenizer example for GPT-4o](https://tiktokenizer.vercel.app/?model=gpt-4o)
:::


#### Step 4: Hyperparameters for Text Generation


Besides the existence of roles, the `pipeline()` method has several hyperparameters (function arguments) for that help us control better how the `text-generation` task will be done. These are some of the most common:

**_max_new_tokens_:** sets maximum number of tokens (roughly words/word pieces) that the model will generate in total. It's a hard limit - generation stops when this limit is reached, even mid-sentence. Useful for controlling cost / time. The more tokens you need to generate for an answer the more time it takes. LLMs called through paid APIs often charge per a set number of tokens (e.g. $0.008 per 1000 tokens).

**_temperature_:** positive float value that controls the randomness/creativity of the model's token selection during generation. The model predicts probabilities for each possible next token, temperature modifies these probabilities before making the final choice. 0.0: Completely deterministic - always picks the most likely token; 1.0+: More random, and "creative", but potentially less coherent.

**_do_sample_:** when do_sample=True, the model generates text by sampling from the probability distribution of possible next tokens. If do_sample=False, the model uses [greedy decoding](https://huggingface.co/docs/transformers/generation_strategies) (always picking the most likely next token), which makes the output more deterministic but often repetitive.

**_top_k_:** This is a sampling strategy called [Top-K sampling](https://arxiv.org/pdf/1805.04833). Instead of considering all possible next tokens, the model looks at the k most likely tokens (based on their probabilities) and samples only from that reduced set. If top_k=50, the model restricts its choices to the top 50 most probable words at each step.

::: challenge
Take 10 minutes to play with the hyperparamenters and observe how they affect outputs, eventhough you are using exactly the same model every time. Try different prompts, messages and values and see if the ouptuts match your intuitions. You can manipulate the values explicitly in the pipeline such as:

```python

messages = [
  {"role": "system", "content": "You are a helpful assistant. Only tell me 'yes' or 'no' and a one-sentence explanation for your answer." },
  {"role": "user", "content": "Is NLP the best research field?" },
]

response = llm(messages, max_new_tokens=100, do_sample=True, top_k=5, temperature=0.7)[0]["generated_text"]
```

Some aspects to discuss:

1. **What did you notice about the models' responses?** 
    - Were they always accurate? Always coherent?
    - How did different prompts affect the quality?

2. **Temperature Effects:**
    - What happened when temperature was low (e.g. 0.0 or 0.1) vs. high (e.g. 1.2)?
    - Under which circumstances would you want more random / creative responses vs. consistent responses?

3. **Max Length Effects:**
    - Did you notice a difference in speed of responses when adjusting the max_length parameter?

:::


## Prompting to Solve NLP Tasks

Now we will link some of the funamentals we learned earlier with the usage of LLMs. As mentioned before, we can consither the LLM as a component that given an input provides an output, thus we can evaluate it as any other Machine Learning classifier. We will re-visit the task of Sentiment Analysis to see how we can shape the task to make the generative model provide the outputs we need. 

But before doing that, we will take the opportunity to introduce [Ollama](https://github.com/ollama/ollama), an open-source tool that allows you to run LLMs locally on your own computer, instead of having to call remote services such as OpenAI, Gemini, etc. Ollama lets you download and execute some of the most popular (semi-)open source models, including embedding models (Encoder-based) and generative models (Chat-Assistants), "thinking" models, etc... Take a look at the [whole list](https://ollama.com/search). Consider that the size of the models you can download and effectively use will depend on how powerful your laptop is. You can use Ollama as a standalone applicaiton in Windows/Mac/Linux, but you can also call its local API using python so you can integrate the LLMs into your scripts. 

We will use Ollama thorugh the [langchain](https://docs.langchain.com/) interface to give you the opportunity to also get to know this popular python library. Using Langchain allows you to write a single piece of code and be flexible to use many closed and open source models and even get production-level code with minimal modifications, so it is nice to try it. This is a very complex library and we will only cover the basics to give you a headstart. We are going to test the [llama3.2:1b](https://ollama.com/library/llama3.2:1b) model which you can download in the Ollama interface, or also from your terminal by running:

```bash
ollama pull llama3.2:1b
```

Let's see how does the `langchain` code looks to get exactly the same kind of interaction as with the HuggingFace code.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
    num_predict=100, # Same as max_new_tokens
    top_k=5, # Only consider the first 5 most likely tokens
    top_p=0.9 # Just consider next tokens form the top_p most likely options. Similar to do_sample
)

# Define messages in LangChain format
messages = [
    ("system", "You are a helpful assistant. Give short straight answers"),
    ("human", "Where is Groningen located?"),
]

# Get response
response = llm.invoke(messages)

# Show the Response object (includes useful metadata stats)
print(response)

# Show content only
print(response.content)
```

Let us now try the sentiment analysis task to see how well different models (with different number of parameters perform). Consider the following set of lines from product reviews:

**Product reviews:**

1. I love this movie! It was absolutely fantastic and made my day. **positive**
2. This product is terrible. I hate everything about it. **negative**
3. Nothing says quality like a phone that dies after 20 minutes. **negative**
4. The movie was dark and depressing — exactly what I was hoping for. **positive**
5. The food was delicious, but the service was painfully slow. **mixed**

Set the prompt for this as (substitute the above sentences for ``{text}`` each time):

``Classify the sentiment of the following text as either POSITIVE or NEGATIVE. Text: "{text}"``

Examine the results afterwards to see which models correctly classified them and which didn't.

```python
sentiment_llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0, # Want to be as deterministic as possible
    num_predict=10, # Keep the answer very short
    top_k=1, # Only consider the next most likely token (Greedy)
)

sentiment_texts = [
    "I love this movie! It was absolutely fantastic and made my day.",
    "This product is terrible. I hate everything about it.",
    "Nothing says quality like a phone that dies after 20 minutes.",
    "The movie was dark and depressing — exactly what I was hoping for.",
    "The food was delicious, but the service was painfully slow."
]

general_prompt = "Classify the sentiment of the following text as either POSITIVE or NEGATIVE. Just return the label, nothing else!"
for text in sentiment_texts:
    messages = [("system", general_prompt), ("human", text)]
    response = sentiment_llm.invoke(messages)
    print(f"Example: {text}")
    print(f"Response: {response.content}")
    print("------")
```

::: challenge

Apply the classifier to the `data/sentiment_film_data.tsv` dataset. Evaluate it using the classification report from scikit-learn. you can reuse the code from lesson 3. 
1. How does the performance compare to the BERT-based classifier?
2. Can you tweak the prompt to improve the results?

:::: solution
Here the code to do the same excercise as lesson 3 but with the generative approach:

```python
from sklearn.metrics import classification_report

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:] # skip header
    sentences, labels = zip(*(line.strip().split('\t') for line in lines))
    return list(sentences), list(labels)

sentiment_llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0, # Want to be as deterministic as possible
    num_predict=10, # Keep the answer very short
    top_k=1, # Only consider the next most likely token (Greedy)
)

sentences, y_true = load_data('data/sentiment_film_data.tsv')

y_pred = []
prompt = "Classify the sentiment of the following text as POSITIVE, NEGATIVE or NEUTRAL. Just return the label, nothing else!"
for sent in sentences:
    messages = [("system", prompt), ("human", sent)]
    response = sentiment_llm.invoke(messages)
    print(sent, response.content)
    y_pred.append(response.content.lower())

print(classification_report(y_true, y_pred))
```

Report:
```output
              precision    recall  f1-score   support

    negative       0.50      1.00      0.67        23
     neutral       0.43      0.08      0.14        37
    positive       0.70      0.82      0.76        40

    accuracy                           0.59       100
   macro avg       0.54      0.64      0.52       100
weighted avg       0.55      0.59      0.51       100
```

::::

:::


### Other NLP tasks

Write and execute the following prompts in Python code within Jupyter notebook.

1. **Question answering:** 
    - answering general knowledge questions
    <br>
    ```
    Human: What is the longest river in the world?
    LLM: The Nile River in Africa is traditionally considered the longest river in the world, stretching about 6,650 km (4,130 miles) through 11 countries before emptying into the Mediterranean Sea.
    ```
    <br>

    - expert advice in a particular domain or scientific field
    <br>
    ```
    Human: What are good strategies for film-making on a budget in Night conditions?
    LLM: Night filmmaking can be both creatively rewarding and technically challenging, especially on a budget. Here are some strategies that indie filmmakers often use to maximize image quality and mood without expensive gear:...
    ```
    <br>

2. **Text generation:**
    - Writing essays, business plans and other documents
    <br>
    ```
    Human: Write a template cover letter for a mid-level software engineering position at a large AI startup.
    LLM: Dear [Hiring Manager’s Name], I am excited to apply for the Software Engineer position at [Company Name]...
    ```
    <br>

    - Creative writing (slogans, poems, fiction, jokes)
    <br>
    ```
    Human: Write a two sentence comedy sketch.
    LLM: [Scene: A man sits in a job interview.] Interviewer: "Your resume says you’re fluent in Python?" Candidate: pulls out a snake from his briefcase "She does most of the talking."
    ```
    <br>

3. **Text summarisation**
    <br>

    ```
    Human: Shorten: "Amsterdam, the capital of the Netherlands, is a city celebrated for its rich history, cultural diversity, and iconic canals. Known as the “Venice of the North,” it is home to a vast network of waterways lined with elegant 17th-century houses, giving the city its distinct charm. Amsterdam seamlessly blends old-world character with modern vibrancy, attracting millions of visitors each year."
    LLM: Amsterdam, the capital of the Netherlands, is known for its history, canals, and 17th-century houses, combining old charm with modern vibrancy that attracts millions each year.
    ```
    <br>

4. **Sentiment or text classification**
<br>

    ```
    Sentiment analysis
    Human: Is this a positive, neutral or negative statement about the movie: "Incredible and not in a good way."
    LLM: That’s a negative statement about the movie. The phrase “Incredible” usually suggests something impressive, but the qualifier “not in a good way” makes it clear the speaker means the film was bad or shockingly poor.
    ```
    <br>

    ```
    Text classification
    Human: Categorise this statement into either sports, business, or science: "Lionel Messi scored twice to lead Argentina to victory in the World Cup qualifier."
    LLM: This statement falls under Sports. It talks about a soccer player and a match result, which clearly relates to athletic competition.
    ```
    <br>

5. **Language translation**
<br>
    ```
    Human: Translate "I have recently naturalised as a Dutch citizen" into Dutch.
    LLM: Ik ben onlangs genaturaliseerd als Nederlands staatsburger.
    ```
    <br>

6. **Generating software code**
<br>

    ```
    Human: Write a short Python function to demonstrate bubble sort
    LLM: ...
    ```
    ```python
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            # Last i elements are already in place
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    # Swap if the element is greater than the next
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    # Example usage
    numbers = [64, 34, 25, 12, 22, 11, 90]
    sorted_numbers = bubble_sort(numbers)
    print(sorted_numbers)
    ```

### Key Takeaways

- **LLMs are generative models** - they predict the next most likely tokens
- **Prompts matter** - the way you ask affects what you get
- **Hyperparameters control behavior** - temperature, max_length, etc. tune the output
- **Models have limitations** - they can be wrong, inconsistent, or biased
- **Size vs Speed trade-off** - smaller models are faster but less capable

---

## Transformers and LLMs

LLMs are also trained using the transformer neural network architecture, making use of the self-attention mechanism discussion in Lesson 02. This means that an LLM is also a transformer-based language model. However, they are distinct from _general_ transformer-based language models in three main characteristics:

1. **Scale:** there are two dimensions in which current LLMs exceed general transformer language models in terms of scale. The most important one is the number of _training parameters_ (weights) that are used for training models. In current models there are hundreds of billions of parameters up to trillions. The second factor is the _amount of training data_ (raw text sequences) used for training. Current LLMs use snapshots of the internet (upwards of hundreds of terabytes in size) as a base for training and possibly augment this with additional manually curated data. The sheer scale characteristic of LLMs mean that such models require extremely resource-intensive computation to train. State-of-the-art LLMs require multiple dedicated Graphical Processing Units (GPUs) with tens or hundreds of gigabytes of memory to load and train in reasonable time. GPUs offer high parallelisability in their architecture for data processing which makes them more efficient for training these models.

2. **Post-training:** After training a base language model on textual data, there is an additional step of fine-tuning for enabling conversation in a prompt style of interaction with users, which current LLMs are known for. After the pre-training and neural network training stages we end up with what is called a _base_ model. The base model is a language model which is essentially a token sequence generator. This model by itself is not suitable for the interaction style we see with current LLMs, which can do things like answer questions, interpret instructions from the user, and incorporate feedback to improve responses in conversations.

3. **Generalization:** LLMs can be applied across a wide range of NLP tasks such as summarization, translation, question answering, etc., without necessarily the need for fine-tuning or training separate models for different NLP tasks.

<img src="fig/llm_analogy3.png" alt="llm engine analogy" width="1000" />

What about the relation between BERT, which we learned about in Lesson 02, and LLMs? Apart from the differences described above, BERT only makes use of the encoder layer of the transformers architecture because the goal is on creating token representations preserving contextual meaning. There is no generative component to do something with those representations.
<br>

<img src="fig/llms_vs_bert2.png" alt="llms vs bert" width="800" />

---

### 2. How are LLMs trained?

Training LLMs involves a series of steps. There are two main phases: pretraining and post training. Pretraining generally involves the following substeps:

#### 2.1 Obtaining and pre-processing textual data for training

- _Downloading and pre-processing text:_ State-of-the-art LLMs include entire snapshots of the internet as the core textual data for training. This data can be sourced from efforts such as [CommonCrawl](https://commoncrawl.org/). Proprietary LLMs may augment or supplement this training data with additional licensed or proprietary textual data (e.g., books) from other sources or companies. The raw web pages are not usable by themselves, we need to extract the raw text from those HTML pages. This requires a preprocessing or data cleaning step.

<img src="fig/html_to_text.png" alt="html to text processing" width="800" />

- _Tokenization:_  As we saw in Lesson 01, the raw text itself cannot be used in the training step, we need a way to tokenize and encode the text for processing by the neural network. As an example of what these encodings look like for OpenAI models like GPT, you can visit [TikTokenizer](https://tiktokenizer.f2api.com/).

<img src="fig/text_to_tokenids.png" alt="tokenization" width="800" />

#### 2.2 Neural network training
With LLMs the training goal is to predict the next token in a one-dimensional sequence of tokens. This is different from BERT where the goal is to predict masked tokens in the input sequence. BERT is therefore not natively developed for generating text, whereas LLMs are. In the internals of the transformer architecture, this is illustrated by the fact that BERT only makes use of the Encoder component to create its contextualised word embeddings. It does not use the Decoder component to generate new tokens for the input sequence.

<img src="fig/llm_training_goal.png" alt="training goal llms" width="800" />

After training we obtain a _base_ LLM which is predicts or generates token sequences that resemble its training data. However, a post training step is required in order to fine-tune the model to accept instructions, answer questions in a conversational style and to have behavior that is more suitable for interaction with humans.

#### 2.3 Post training
What does post training for LLMs look like? Why is this step necessary? What would happen if you skip this step and just use the base model trained in Step 2.2 for inference? The answer is that the base model is just a token sequence predictor. It just predicts the most likely next token for an input sequence of tokens. It does not understand how to deal with conversations or to interpret instructions (the intentions and tone behind written communication).

Therefore, you may encounter unexpected conversations like this if interacting with a base LLM:

**Not Following Instructions**

```
Human: Summarize this paragraph in one sentence: The Nile is the longest river in Africa and flows through 11 countries before emptying into the Mediterranean Sea.
Raw LLM: The Nile is the longest river in Africa and flows through 11 countries before emptying into the Mediterranean Sea.
```

In this example interaction, the model was trained to predict text, not to follow instructions. So it might not give expected or correct responses although, statistically, these response tokens are indeed the next most likely tokens.

**Regurgitation**

```
Human: Donald John Trump (born June 14, 1946) is an American politician,
Raw LLM: media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021...
```
<img src="fig/trump.png" alt="tokenization" width="800" />

In this interaction, the model just "regurgitates" verbatim a response from Wikipedia (part of it's training data), instead of creatively formulating a response in its own words. These are just a few examples of the kind of undesirable behavior one might encounter when doing inference with the base model without post training.

Therefore, to avoid issues like this, and to be useful as conversational assistants, LLMs generally have to be fine-tuned on additional unseen conversation datasets (containing on the order of hundreds of thousands of conversations). These datasets are manually created and validated by humans. For proprietary models, these humans are hired professionally to contribute to this dataset from platforms such as Upwork and ScaleAI. There are also open-source efforts to create such datasets such as [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst2).

Once such a dataset is obtained, the conversations need to be represented as a one-dimensional token sequence. Conversations therefore have to be encoded into such representations before being used as input for training. The encoding method for conversations currently differ among LLMs. The figure below shows how GPT-4 tokenizes and encodes input conversations as one dimensional token sequences ([TikTokenizer](https://tiktokenizer.f2api.com/) is a page for visualizing and experimenting with tokenization approaches used by OpenAI LLMs).

<img src="fig/tokenize_conversations.png" alt="tokenization for conversation data" width="800" />

After encoding or tokenization, the conversational data is supplied as input for fine-tuning the base model (replacing the original training data). I.e., the base model's weights or parameters are updated through this process.

Below are some examples of conversational data that one might use to fine-tune an LLM. In general, you want multi-turn conversations where each participant in the conversation gets the opportunity to respond multiple times. Because this is the likely mode of interaction that will take place between humans and LLM-based chat assistants:

```
**Conversation 1 (Travel Planning):**
Human: Can you suggest a weekend getaway near Amsterdam?
LLM: Sure! Giethoorn – Known as the “Venice of the North,” this village has canals, thatched-roof houses, and a peaceful atmosphere. Utrecht, a charming city with medieval streets, beautiful canals, and cozy cafés.
Human: Which one would be better for a quiet, relaxing trip?
LLM: Giethoorn would be the quieter option. Unlike Utrecht, which is a bustling city, Giethoorn is a small village with a calm, serene atmosphere, ideal for a peaceful getaway.

**Conversation 2 (Math Help):**
Human: What's the derivative of 3𝑥^2 + 5𝑥 − 7?
LLM: The derivative is 6𝑥 + 5.
Human: Great, and what about the second derivative?
LLM: The second derivative is 6.

**Conversation 3 (Creative Writing):**
Human: Give me a one-line horror story.
LLM: The reflection in the mirror smiled back before I did.
Human: Can you make it even creepier?
LLM: The reflection winked, though I was alone in the room.
```
---

## Drawbacks and Biases with LLMs

See [notebooks/responsible.ipynb]()...

## Advanced Material

### LLM selection criteria

Choosing the right LLM for your specific use case requires consideration of multiple factors. This section will guide you through some decision points that will help you select an appropriate model for your needs.

#### 1.3.1 Openness and Licensing Considerations

The spectrum of model availability ranges from fully open to completely proprietary:

**Open-weights** release the trained model parameters while keeping training code or data proprietary. This allows you to run and fine-tune the model locally but if you don't have the code used to train the model or information about the architecture used, it limits your ability to fully understand or replicate the training process.

**Open training data** they release the text data used for pretraining.

**Open architecture** they publish a paper about the neural network architecture and specific configuration they used for training. Or they release the actual source code they used for pretraining.

Ideally, if you want to use a model for empirical academic research you might decide for models that are completely open in all three of the above facets. Although, open training data is quite rare for available state-of-the-art models.

**Commercial/proprietary models** like GPT-4, Claude, or Gemini are accessed only through APIs. While often offering superior performance, they provide no access to internal architecture and may have usage restrictions or costs that scale with volume.

Consider your requirements for:
- Code modification and customization
- Data privacy and control
- Commercial usage rights
- Research reproducibility
- Long-term availability guarantees

If you wish to build an application that makes use of LLM text generation, and you need accurate results, commercial APIs may be more suitable.

#### 1.3.2 Hardware and Compute Requirements

Your available computational resources significantly constrain your model options:

**Modern GPU access** (RTX 4090, A100, H100, etc.) enables you to run larger models locally. Consider:
- VRAM requirements: 7B parameter models typically need 14+ GB, 13B models need 26+ GB, 70B models require 140+ GB or multi-GPU setups
- Inference speed requirements for your application
- Whether you need real-time responses or can accept slower processing

**CPU-only environments** limit you to smaller models (such as SmolLM2 and SmolLM3) or [quantized](https://ojs.aaai.org/index.php/AAAI/article/view/29908) versions.

**Cloud/API access** removes hardware constraints but introduces ongoing costs and potential latency issues.

#### 1.3.3 Performance Evaluation

Different models excel at different tasks. Some evaluation criteria include:

**General capability benchmarks** like those found on the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) provide standardized comparisons across models for reasoning, knowledge, and language understanding tasks.

**Multilingual performance** varies significantly between models. The [MMLU-Pro benchmark](https://mmluprox.github.io/) offers insights into cross-lingual capabilities if you need support for non-English languages.

**Task-specific performance** should be evaluated based on your particular needs:
- Code generation
- Mathematical reasoning
- Reading comprehension and summarization
- Creative writing and dialogue quality
- Scientific and technical domain knowledge

Always validate benchmark performance with your own test cases, as real-world performance may differ from standardized evaluations.

#### 1.3.4 Purpose or Use Case

**Scientific and research applications** often prioritize reproducibility, transparency, and the ability to modify model behavior. Open-source models with detailed documentation are typically preferred (e.g. SmolLM, LLama, Olmo)

**Applications (mobile or web apps)** may require:
- Reliable API uptime and support
- Clear licensing for commercial use
- Scalability to handle many concurrent users
- Content filtering and safety features

**Personal or educational use** might emphasize:
- Cost-effectiveness
- Ease of setup and use

#### 1.3.5 Integration and Deployment Considerations

**Software integration** requirements affect model choice:
- API-based models offer simpler integration but require internet connectivity
- Local models provide more control but require more complex deployment
- Consider latency requirements, offline capabilities, and data privacy needs

**Hosting and serving capabilities** determine whether you can run models locally:
- Do you have the infrastructure to serve models at scale?
- Are you self-hosting the model?

#### 1.3.6 Domain-Specific Models

Many models have been fine-tuned for specific domains or tasks. For example:

- Medical and healthcare applications (e.g., [BioGPT](https://huggingface.co/microsoft/biogpt))
- Legal document processing (e.g., [SaulLM](https://huggingface.co/Equall/Saul-7B-Instruct-v1))

Remember that the LLM landscape evolves rapidly. New models are released frequently, and performance benchmarks should be regularly reassessed. Consider building your system with model-agnostic interfaces to facilitate future transitions between different LLMs as your needs evolve or better options become available.

---

### 3. LLM Prompting strategies

See [notebooks/prompting.ipynb]()...

---

### 4. Solving NLP tasks with LLMs

See [notebooks/classifandeval.ipynb]()...

---

### 5. Creating your own chat assistant

See [notebooks/chatbot.ipynb]()...

---
