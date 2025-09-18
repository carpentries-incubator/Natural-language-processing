---
title: "Episode 3: Using large language models"
teaching: 
exercises: 
---

:::::: questions 
1. What is the relation between Transformers and "LLM's"? [**partially**]

2. Relation between BERT and LLMs? [**partially**]

3. What kinds of tasks can LLMs solve? How can NLP help? [**partially**]

4.  What are the criteria for selecting LLMs to use? What aspects to consider? [**draft**]

5. What are some common prompt strategies? [**Not yet**]

6. How to use HuggingFace to use local LLMs for inference? [**TODO**]

7. How to use LLMs Responsibly? (drawbacks, errors and biases) [**Not yet**]

8. How to build a simple "Chatbot"? -> try to use command input in Jupyter itself instead of command line [**TODO**]

9. How can we evaluate the responses that we get? [**Not yet**]

10. Could mention Ethics and sustainability of training models? [**Not yet**]

11. Make finer-grained distinction between open-source models (open-weights, open training data, open ...?)
- Add AllenAI Olmo (full open-source) [**done**]
- GPT-NL [**done**]

12. For the application section can already start with coding (loading and doing inference in the notebook)
- Can do the first two tasks and let them do as an exercise the remainder of the tasks? [**doneintro**]

13. Too much detail in post training section. Angel will reduce detail in lesson 02. 

14. Comparison of LLMs:
    - Text classification tasks of increasing difficulty and see where small ones fail and large one solves them.



::::::

:::::: objectives
After following this lesson, learners will be able to:

1. Explain the relation between Transformer architecture and LLMs
2. Judge which LLMs are most suitable for a particular task
3. Explain how LLMs can be used to solve established NLP tasks
4. Evaluate the efficacy of LLMs to solve common NLP tasks
5. Use Python and HuggingFace to load and employ LLMs to perform basic inference
6. Explain what Retrieval Augmented Generation (RAG) is and its relation to LLMs
7. Understand the impact of LLMs in modern AI and language processing

**To do:**

1. Diagram transformers vs. LLMs (post-training step)
2. BERT diagram showing different between BERT and LLMs [easy]
3. Input in Jupyter notebooks for chat style interaction [easy]


## Background

Chat assistants like [ChatGPT](https://chatgpt.com/) and [Claude](http://claude.ai), which are based on Large Language Models (LLMs) are widely used today for tasks such as content generation, question answering, research and software development. The rapid rise of such models has had quite a disruptive and strong impact. But what are these models exactly? How do they work 'under the hood'? And how can one use them programmatically, in a responsible and effective way?

This episode is a gentle introduction to LLMs which aims to equip you with knowledge of the underpinnings of LLMs based on transformers architecture, as well as practical skills to programmatically work with LLMs in your own projects.

<img src="fig/llm-logos/anthropic.png" alt="Company A" width="80"/> <img src="fig/llm-logos/alibaba.png" alt="Company B" width="80"/> <img src="fig/llm-logos/xai.jpg" alt="Company C" width="80"/> <img src="fig/llm-logos/zhipu.png" alt="Company C" width="150"/> <img src="fig/llm-logos/google.png" alt="Company C" width="80"/> <img src="fig/llm-logos/openai.jpg" alt="Company C" width="150"/>

<img src="fig/llm-logos/nvidia.png" alt="Company D" width="80"/> <img src="fig/llm-logos/deepseek.png" alt="Company E" width="80"/> <img src="fig/llm-logos/huggingface.png" alt="Company F" width="80"/> <img src="fig/llm-logos/meta.png" alt="Company C" width="150"/>

## 1. What are Large Language Models (LLMs)?
Large language models (LLMs) are transformer-based language models that are specialised to interpret and generate text, and to converse in a conversational-like manner with humans. The text that they generate are mostly natural language but can, in theory, constitute any character or symbol sequence such as software code. They represent a significant advancement in AI and NLP. and are trained on vast amounts of textual data mostly obtained from the internet.

### 1.1 Examples of LLMs 

Many different LLMs have been, and continue to be, developed. There are both proprietary and open-source varieties. Open-source varieties often make the data that their LLMs are trained on free, open and accessible online. Some even make the code they use to train these models open-source as well. Below is a summary of some current LLMs together with their creators, chat assistant interfaces, and proprietary status:

<img src="fig/llm_table4.png" alt="LLMs table" width="1000" />

### 1.2 Applications of LLMs

LLMs can be used for many different helpful tasks. Some common tasks include:

- Question Answering
- Text Generation
- Text Summarisation
- Sentiment Analysis
- Machine Translation
- Code Generation

#### Exercise 1: Your first programmatic LLM interaction (30 minutes)

Before exploring how we can invoke LLMs programmatically to solve the kinds of tasks abve, let us setup and load our first LLM.

##### Step 1. Setup code
Install required packages ``transformers`` and ``torch`` and import required libraries.
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
```

##### Step 2: Load and setup an LLM

Let's load a lightweight LLM.

```python
# We'll use SmolLM-135M - an open, small, fast model
# model_name = "HuggingFaceTB/SmolLM2-135M" # base model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct" # fine-tuned assistant model
# model_name = "HuggingFaceTB/SmolLM3-3B-Base" # base model
# model_name = "HuggingFaceTB/SmolLM3-3B" # fine-tuned assistant model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if model is loaded correctly
print(f"Model loaded! It has {model.num_parameters():,} parameters")
```

##### Step 3: Basic Text Generation

Let's perform inference with the LLM to generate some text.

```python
# Set pad_token_id to eos_token_id to avoid warnings
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.eos_token_id
    
# Build pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "{}"
response = llm(prompt, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)[0]["generated_text"]
print(f"Prompt: {prompt}")
print(f"Response: {response}")
```
**_max_new_tokens_:** sets maximum number of tokens (roughly words/word pieces) that the model will generate in total. It's a hard limit - generation stops when this limit is reached, even mid-sentence. Useful for controlling cost / time. The more tokens you need to generate for an answer the more time it takes. LLMs called through paid APIs often charge per a set number of tokens (e.g. $0.008 per 1000 tokens).

**_temperature_:** positive float value that controls the randomness/creativity of the model's token selection during generation. The model predicts probabilities for each possible next token, temperature modifies these probabilities before making the final choice.

0.0: Completely deterministic - always picks the most likely token
1.0+: More random, and "creative", but potentially less coherent

**_do_sample_:** when do_sample=True, the model generates text by sampling from the probability distribution of possible next tokens. If do_sample=False, the model uses [greedy decoding](https://huggingface.co/docs/transformers/generation_strategies) (always picking the most likely next token), which makes the output more deterministic but often repetitive.

**_top_k_:** This is a sampling strategy called [Top-K sampling](https://arxiv.org/pdf/1805.04833). Instead of considering all possible next tokens, the model looks at the k most likely tokens (based on their probabilities) and samples only from that reduced set. If top_k=50, the model restricts its choices to the top 50 most probable words at each step.

##### Step 5. Sentiment analysis
Let us try a sentiment analysis task to see how well different models (with different number of parameters perform). Consider the following set of lines from product reviews:

**Product reviews:**

1. I love this movie! It was absolutely fantastic and made my day. [**positive**]
2. This product is terrible. I hate everything about it. [**negative**]
3. Nothing says quality like a phone that dies after 20 minutes. [**negative**]
4. The movie was dark and depressing — exactly what I was hoping for. [**positive**]
5. The food was delicious, but the service was painfully slow. [**mixed**]

Set the prompt for this as (substitute the above sentences for ``{text}`` each time):

``Classify the sentiment of the following text as either POSITIVE or NEGATIVE. Text: "{text}"``

Examine the results afterwards to see which models correctly classified them and which didn't.

```python
sentiment_llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
sentiment_texts = [
    "I love this movie! It was absolutely fantastic and made my day.",
    "This product is terrible. I hate everything about it.",
    "Nothing says quality like a phone that dies after 20 minutes.",
    "The movie was dark and depressing — exactly what I was hoping for.",
    "The food was delicious, but the service was painfully slow."
]
text = sentiment_texts[0]
prompt = "Classify the sentiment of the following text as either POSITIVE or NEGATIVE. Text: "{text}""
response = sentiment_llm(prompt, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)[0]["generated_text"]
print(f"Prompt: {prompt}")
print(f"Response: {response}")
```

##### Discussion: Post-exercise questions

1. **What did you notice about the models' responses?** 
   - Were they always accurate? Always coherent?
   - How did different prompts affect the quality?

2. **Temperature Effects:**
   - What happened when temperature was low (e.g. 0.0 or 0.1) vs. high (e.g. 1.2)?
   - Under which circumstances would you want more random / creative responses vs. consistent responses?

3. **Model Size:**
   - What were the differences across different models?
   - What trade-offs do you think exist between model size and performance?

4. **Max Length Effects:**
   - Did you notice a difference in speed of responses when adjusting the max_length parameter?

#### Exercise 2: Other NLP tasks

<!-- **Prompt Engineering Preview:**
   - Which prompts gave better results? What made them effective? -->

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

```
Human: Shorten: "Amsterdam, the capital of the Netherlands, is a city celebrated for its rich history, cultural diversity, and iconic canals. Known as the “Venice of the North,” it is home to a vast network of waterways lined with elegant 17th-century houses, giving the city its distinct charm. Amsterdam seamlessly blends old-world character with modern vibrancy, attracting millions of visitors each year."
LLM: Amsterdam, the capital of the Netherlands, is known for its history, canals, and 17th-century houses, combining old charm with modern vibrancy that attracts millions each year.
```
<br>

4. **Sentiment or text classification**

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

```
Human: Translate "I have recently naturalised as a Dutch citizen" into Dutch.
LLM: Ik ben onlangs genaturaliseerd als Nederlands staatsburger.
```
<br>

6. **Generating software code**
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




### Part 3: Experimenting with Different Prompts (5 minutes)

```python
# Let's try different types of tasks to see what our LLM can do
prompts_to_try = [
    "Write a haiku about coding:",
    "Translate 'Hello, world!' to French:",
    "Complete this code: def fibonacci(n):",
    "What are the benefits of using Python?",
    "Write a short story about a robot learning to code:",
]

print("=== Trying Different Prompts ===")
for prompt in prompts_to_try:
    response = generate_response(prompt, max_length=80)
    print(f"\n🔸 Prompt: {prompt}")
    print(f"📝 Response: {response[len(prompt):].strip()}")  # Remove the prompt from output
    print("-" * 50)
```

### Part 4: Understanding Generation Parameters (5 minutes)

```python
# Let's see how different parameters affect the output
test_prompt = "The best programming language is"

print("=== Effect of Temperature on Generation ===")
temperatures = [0.1, 0.7, 1.2]

for temp in temperatures:
    print(f"\n🌡️ Temperature: {temp}")
    for i in range(3):  # Generate 3 examples for each temperature
        response = generate_response(test_prompt, max_length=50, temperature=temp)
        clean_response = response[len(test_prompt):].strip()
        print(f"   {i+1}. {clean_response}")
```

### Part 5: Using Pipeline API (Bonus - 2 minutes)

```python
# Hugging Face also provides a simpler pipeline API
print("\n=== Using Pipeline API ===")
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=model_name,
    device=0 if device == "cuda" else -1
)

# This is often easier for simple use cases
simple_response = generator(
    "Python is a great language because",
    max_length=60,
    num_return_sequences=2,
    temperature=0.8
)

for i, response in enumerate(simple_response, 1):
    print(f"{i}. {response['generated_text']}")
```

## Discussion Questions (to be covered after coding)

1. **What did you notice about the model's responses?** 
   - Were they always accurate? Always coherent?
   - How did different prompts affect the quality?

2. **Temperature Effects:**
   - What happened when temperature was low (0.1) vs high (1.2)?
   - When might you want creative vs consistent responses?

3. **Model Size:**
   - We used a 135M parameter model. What trade-offs do you think exist between model size and performance?

4. **Prompt Engineering Preview:**
   - Which prompts gave better results? What made them effective?

## Key Takeaways

- **LLMs are generative models** - they predict the next most likely tokens
- **Prompts matter** - the way you ask affects what you get
- **Parameters control behavior** - temperature, max_length, etc. tune the output
- **Models have limitations** - they can be wrong, inconsistent, or biased
- **Size vs Speed trade-off** - smaller models are faster but less capable

## What's Next?

In the rest of our lesson, we'll explore:
- How to craft better prompts (prompt engineering strategies)
- How to choose the right model for your task
- Methods for comparing and evaluating different LLMs
- Real-world applications and best practices

---

**Troubleshooting:**
- If you get memory errors, try using a smaller model like "microsoft/DialoGPT-small"
- If generation is slow, reduce max_length or use CPU instead of trying GPU
- If outputs seem repetitive, try adjusting temperature or adding `do_sample=True`

### 1.3 LLM selection criteria

- Open-source vs. Proprietary?
- Available compute: do you have a modern GPU or not?
- Multilingual performance? https://mmluprox.github.io/
- Other task performances? https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/
- Do you have a scientific, commercial or personal purpose for using an LLM?
- Do you want to integrate an LLM into a software application? 
- Do you have resources to host or serve an LLM?
- Fine-tuned models on different domains. Embedding models, buzzword list for fine-tuned models (e.g. thinking)


### 1.4 Transformers and LLMs

LLMs are also trained using the transformer neural network architecture, making use of the self-attention mechanism discussion in Lesson 02. This means that an LLM is also a transformer-based language model. However, they are distinct from _general_ transformer-based language models in three main characteristics:

1. **Scale:** there are two dimensions in which current LLMs exceed general transformer language models in terms of scale. The most important one is the number of _training parameters_ (weights) that are used for training models. In current models there are hundreds of billions of parameters up to trillions. The second factor is the _amount of training data_ (raw text sequences) used for training. Current LLMs use snapshots of the internet (upwards of hundreds of terabytes in size) as a base for training and possibly augment this with additional manually curated data. The sheer scale characteristic of LLMs mean that such models require extremely resource-intensive computation to train. State-of-the-art LLMs require multiple dedicated Graphical Processing Units (GPUs) with tens or hundreds of gigabytes of memory to load and train in reasonable time. GPUs offer high parallelisability in their architecture for data processing which makes them more efficient for training these models.

2. **Post-training:** After training a base language model on textual data, there is an additional step of fine-tuning for enabling conversation in a prompt style of interaction with users, which current LLMs are known for. After the pre-training and neural network training stages we end up with what is called a _base_ model. The base model is a language model which is essentially a token sequence generator. This model by itself is not suitable for the interaction style we see with current LLMs, which can do things like answer questions, interpret instructions from the user, and incorporate feedback to improve responses in conversations.

3. **Generalization:** LLMs can be applied across a wide range of NLP tasks such as summarization, translation, question answering, etc., without necessarily the need for fine-tuning or training separate models for different NLP tasks.

<img src="fig/llm_analogy3.png" alt="llm engine analogy" width="1000" />

What about the relation between BERT, which we learned about in Lesson 02, and LLMs? Apart from the differences described above, BERT only makes use of the encoder layer of the transformers architecture because the goal is on creating token representations preserving contextual meaning. There is no generative component to do something with those representations.
<br>

<img src="fig/llms_vs_bert2.png" alt="llms vs bert" width="800" />

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

### 3. Creating your own chat assistant
Let's interact with LLM models through Python code and create our own basic chat assistant. We are going to use existing pre-trained models from [HuggingFace](https://huggingface.co/).

Lets first setup code to load the LLMs using the [transformers](https://github.com/huggingface/transformers)

```python
# Import required libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# just so we ignore annoying messages in the console from Jupyter
import warnings 
warnings.filterwarnings("ignore")

# Pick a model
model_id = "HuggingFaceTB/SmolLM2-135M" # base model
# model_id = "HuggingFaceTB/SmolLM2-135M-Instruct" # fine-tuned assistant model
# model_id = "HuggingFaceTB/SmolLM3-3B-Base" # base model
# model_id = "HuggingFaceTB/SmolLM3-3B" # fine-tuned assistant model
# model_id = "meta-llama/Llama-3.2-3B-Instruct" # fine-tuned assistant model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

Lets also add some code to disable more warning messages which can clutter the console:

```python
# Set pad_token_id to eos_token_id to avoid warnings
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
```

Let's now create and initialise our chat assistant with the model and tokenizer:

```python
# Build pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

We will now define a convenience function for creating a chat prompt (input):

```python
# This variable will store our conversation history with the chat assistant
history = []

def chat(user_input):
    global history
    history.append(f"User: {user_input}")
    prompt = "\n".join(history) + "\nAssistant:"
    
    response = chatbot(prompt, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)[0]["generated_text"]
    
    # Extract assistant’s reply after the last "Assistant:"
    reply = response.split("Assistant:")[-1].strip()
    history.append(f"Assistant: {reply}")
    return reply
```

Now let us actually run our chat assistant! We create a loop to ensure that the conversation continues:

```python
# Call chatbot interaction
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    print("Bot:", chat(user_input))

```
#### 3.1 Comparing different LLMs
...

#### 3.2 Prompt strategies
...

#### 3.3 Drawbacks, Errors and Biases
LLMs have strengths and weaknesses (although the weaknesses are improving rapidly). If not intentionally addressed during training, LLMs can display the following examples of undesirable behavior:

**Hallucination**

Load and initialise the ``HuggingFaceTB/SmolLM3-3B`` model in your notebook. Run the following code directly after (create a new cell if necessary):

```python
halluc_prompt = "Who is Railen Ackerby?"
infengine = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = infengine(halluc_prompt, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)[0]["generated_text"]
print(response.strip())
```

You may get a response like: "Who is Railen Ackerby? How did he get his name? Railen Ackerby is an English musician and composer. He was born in Birmingham, England in 1992. Railen Ackerby’s name is a combination of Railen, his mother’s maiden name, and Ackerby, the last name of his paternal grandfather. Railen’s paternal grandfather was an English composer and musicologist who played the violin and studied at Oxford University. Railen was named after his grandfather."

However, this person does not actually **exist**. So the model is "hallucinating" or "seeing something that is not there" i.e., making something up.

**Biases or stereotypes**

LLMs may produce biased or offensive text based on biases present in the training data, making content moderation necessary in sensitive applications.

```python
bias_prompt = "Write a two paragraph story where a nurse, a pilot, and a CEO are having lunch together."
response = infengine(bias_prompt, max_new_tokens=500, do_sample=True, top_k=50, temperature=0.7)[0]["generated_text"]
print(response.strip())
```

You may get a response similar to (you may not get exactly the same output due to the stochastic processes which define how these models work):

"In a bustling city, amidst the hum of traffic and the chatter of pedestrians, three individuals gathered for lunch in a cozy café. Dr. Emma Taylor, a dedicated nurse with a heart full of compassion, sipped _**her**_ green tea, _**her**_ eyes gleaming with warmth as she observed the world around _**her**_"

This is of course reinforcing gender stereotypes for the nurse profession. Even the most advanced models are susceptible to these biases and stereotypes.

### Model architecture
- LLMs: Use the Transformer architecture, particularly self-attention, to analyze relationships between words regardless of position. This allows them to capture long-range dependencies and context better than traditional models.
- Traditional NLP: Often use simpler models like bag-of-words, TF-IDF (term frequency-inverse document frequency), RNNs (recurrent neural networks), and LSTMs (long-short-term memory models),  which treat words independently or consider only local context, missing the complex, global relationships.

### Learning from unlabeled data
- LLMs: Leverage unsupervised or self-supervised learning during pretraining, enabling them to learn language patterns from raw text without human-labeled data.
- Traditional NLP: are often supervised models, relying on labeled data for training (e.g., labeled sentiment or part-of-speech tags), which can be costly and time-consuming to create at scale.

### Adaptability and fine-tuning
- LLMs: Easily adaptable to new tasks or domains with fine-tuning, making them versatile across different applications.
- Traditional NLP: Less flexible, often requiring retraining from scratch or heavy feature engineering to adapt to new domains or tasks.

## What LLMs are good at
- *Language generation*: Creating coherent and contextually appropriate text, making them ideal for creative writing, chatbots, and automated responses.
- *Summarization and translation*: Quickly summarizing articles, books, and translating text between languages with reasonable accuracy.
- *Information retrieval and answering questions*: LLMs can recall and apply general knowledge from their training data to answer questions, though they don’t actually “know” facts.
- *Sentiment and text classification*: LLMs can classify text for tasks like sentiment analysis, spam detection, and topic categorization.

## What LLMs struggle with
- *Fact-based accuracy*: Since LLMs don’t “know” facts, they may generate incorrect or outdated information and are prone to hallucinations (making up facts).
- *Understanding context over long passages*: LLMs can struggle with context over very long texts and may lose track of earlier details, affecting coherence.
- *Mathematical reasoning and logic*: Though improving, LLMs often find complex problem-solving and detailed logical reasoning challenging without direct guidance.
- *Ethical and sensitive issues*: LLMs may produce biased or offensive text based on biases present in the training data, making content moderation necessary in sensitive applications.

- *Transformers and self-attention*: The transformer architecture, especially the self-attention mechanism, is at the heart of LLMs. Self-attention enables these models to understand the importance of each word in relation to others in a sequence, regardless of their position.
- *Pretraining and fine-tuning*: LLMs are first pre-trained on large text datasets using tasks like predicting the next word in a sentence, learning language patterns. They are then fine-tuned on specific tasks (e.g., translation, summarization) to enhance performance for targeted applications.
- *Generative vs. discriminative models*: LLMs can be applied to both generative tasks (e.g., text generation) and discriminative tasks (e.g., classification).





In practice, this attention mechanism helps LLMs produce coherent responses by establishing relationships between words as each new token is generated. Here’s how it works: 

- *Understanding word relationships*. Self-attention enables the model to weigh the importance of each word in a sentence, no matter where it appears, to make sense of the sentence as a whole.

- *Predicting next words based on context*. With these relationships mapped out, the model can predict the next word in a sequence. For example, in “The fox,” self-attention allows the model to anticipate that “jumps” or “runs” might come next rather than something unrelated like “table.”

- *Structuring responses*. As each word is generated, the model assesses how each new token impacts the entire sentence, ensuring that responses are relevant, logically sound, and grammatically correct. This ability to “structure” language is why LLMs can produce responses that are contextually meaningful and well-organized.
  


Training a large language model is extremely resource intensive. For example, llama's model Llama 3.1 405B is a model that has 405 billion parameters. It was trained on 15 trillion tokens, uses 31 million GPU hours (H100 gpus), and emitted almost 9000 tons of CO_2 (for the training process only).

Inference also consumes considerable resources and has a significant environmental impact. Large models require large memory for storing and loading the model weights (storing weights alone can require _hundreds_ of gigabytes), and need high-performance GPUs to achieve reasonable runtimes. As a result, many models operate on cloud-based servers, increasing power consumption, especially when scaled accomodate large numbers of users.

## Which one to chose when?

With so many available models the question arises "which model you should use when"? One thing to consider here is whether you want to use an open source model or not. But another important aspect is that it depends on the task at hand. There are various leaderboards (for example: [HuggingFace](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/, [HELM](https://crfm.stanford.edu/helm/lite/latest/)) that track which tasks specific models are good at, based on widely used benchmarks. Also, which language are you using? Most models are fully trained on English, not many models are trained on Dutch text. So if you are using Dutch texts, you may want to look for a model that is trained on or finetuned for Dutch. Additionally, some LLMs are multimodal models, meaning they can process various forms of input; text, images, timeseries, audio, videos and so on.

## Building a chatbot
It is time to start using an LLM! We are not going to train our own LLM, but use Meta's open source Llama model to set up a chatbot. 

#### Starting Ollama
Ollama is a platform that allows users to run various LLM locally on your own computer. This is different from for example using chatgpt, where you log in and use the online api. ChatGPT collects the input you are providing and uses this to their own benefit. Running an LLM locally using Ollama thus preserves your privacy. It also allows you to customize a model, by setting certain parameters, or even by finetuning a model. 

To start Ollama:
```
ollama serve
```

Next, download the large language model to be used. In this case use the smallest open source llama model, which is llama3.1:8b. Here 3.1 is the version of the model and 8b stands for the number of parameters that the model has. 
```
!ollama pull llama3.1:8b
```
In general, a bigger version of the same model (such as Llama3.1:70b) is better in accuracy, but since it is larger it takes more resources to run and can hence be too much for a laptop.

Import the packages that will be used:
```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
```

### Create a model instance

Here, `model` defines the LLM to be used, which is set to the model just downloaded, and `temperature` sets the randomness of the mode, using the value zero ensures that repeating a question will give the same model output (answer).
```
llm = ChatOllama(model="llama3.1:8b", temperature=0)
```

Now that the model is set up, it can be invoked - ask it a question.

```python
question = "When was the moon landing?"
chatresult = llm.invoke([HumanMessage(content=question)])
print(chatresult.content)
```

:::::::::::: challenge 

Play around with the chat bot by changing the questions.
- How is the quality of the answers? 
- Is it able to answer general questions, and very specific questions?
- Which limitations can you identify?
- How could you get better answers?

:::::: solution

::::::

This Llama chat bot, just like ChatGPT, is quite generic. It is good at answering general questions; things that a lot of people know. Going deeper and asking very specific questions often leads to vague or inaccurate results. 

::::::::::::


### Use context
To improve on what to expect the LLM to return, it is also possible to provide it with some context. For example, add:
```python
context = "You are a highschool history teacher trying to explain societal impact of historic events."
messages = [
    SystemMessage(content=context),
    HumanMessage(content=question),
]
```

```python
chatresult = llm.invoke(messages)
print(chatresult.content)
```

The benefit here is that your answer will be phrased in a way that fits your context, without having to specify this for every question.

### Use the chat history
With this chatbot the LLM can be invoked to generate output based on the provided input and context. However, what is not possible in this state, is to ask followup questions. This can be useful to refine the output that it generates. The next step is therefore to implement message persistence in the workflow.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from IPython.display import Image, display
```

The package LangGraph is a library that is designed to build LLM agents using workflows represented as graphs. The workflows you create consist of connected components, which allows you to build multi-step processes. The workflow graphs can be easily visualised which makes them quite insightful. LangGraph also has a build-in persistence layer, exactly what we want right now!

First, define an empty workflow graph with the StateGraph class with the MessageState schema (a simple schema with messages as only key)
```python
workflow = StateGraph(state_schema=MessagesState)
```

Then define a function to invoke the llm with a message

```python
def call_llm(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}
```

Then add the call_llm function as a node to the graph and connect it with an edge to the start point of the graph. This start node sends the user input to the graph, which in this case only contains the LLM element.

```python
workflow.add_node("LLM", call_llm)
workflow.add_edge(START, "LLM")
```

Initialise a memory that will preserve the messages state in a dictionary while going though the graph multiple times asking followup questions.
```python
memory = MemorySaver()
```

Then compile and visualise the graph with the memory as checkpoint.
```python
graph = workflow.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))
```

![workflow](./fig/workflow_llm.png)
Define a memory id for the current conservation.

```python
config = {"configurable": {"thread_id": "moonconversation"}}
```

Then call the workflow with memory we created with the original question
```python
question = 'Who landed on the Moon?'
messages = [HumanMessage(question)]
output = graph.invoke({"messages": messages}, config)

output["messages"][-1].pretty_print()
```

The question and answer are now saved in the graph state with this config, and followup questions and answers with the same config will be added to it.

Everything that is saved can be found in the config state
```python
graph.get_state(config)
```

The workflow can now be used to ask followup questions without having to repeat the original question, and based on the previous generated answer.
```python
# Followup
followup = "Shorten the answer to 20 words"
input_messages = [HumanMessage(followup)]
output = graph.invoke({"messages": input_messages}, config)

# print the last output
output["messages"][-1].pretty_print()
```


```python
# Followup instruction
followup2 = "Translate the answer to Dutch"
input_messages = [HumanMessage(followup2)]
output = graph.invoke({"messages": input_messages}, config)

# print the last output
output["messages"][-1].pretty_print()
```

## Retrieval Augmented Generation - Build a RAG
A chatbot tends to give quite generic answers. A more specific chatbot can be made by building a Retrieval Augmented Generation agent. This is an information that you yourself provide with a knowledge base: a large number of documents. When prompted with a questions, the agent first retrieves relevant sections of the data that is in the knowledge base, and then generates and answer based on that data. In this way you can build an agent with very specific knowledge.

The simplest form of a rag consists of two parts, a retriever and a generator. The retriever part will collect data from the provided data, so first a knowledge base has to be created for the retriever.

To generate text in the RAG the trained Llama model will be used, which works well for English text. Because this model was not trained on Dutch text, the RAG will work better for an English knowledge base.

Three newspaper pages will be used for the example RAG, these are pages from a Curacao newspaper. This is a Dutch newspaper with an additional page in English. The text versions of the newspapers can be downloaded to only get these specific English pages. Save them in a folder called "rag_data" for further processing:
- [page1](https://www.delpher.nl/nl/kranten/view?query=the+moon&coll=ddd&identifier=ddd:010460545:mpeg21:p012&resultsidentifier=ddd:010460545:mpeg21:a0134&rowid=4)
- [page2](https://www.delpher.nl/nl/kranten/view?query=moon+landing&coll=ddd&page=1&facets%5Bspatial%5D%5B%5D=Nederlandse+Antillen&identifier=ddd:010460616:mpeg21:a0146&resultsidentifier=ddd:010460616:mpeg21:a0146&rowid=1)
- [page3](https://www.delpher.nl/nl/kranten/view?query=moon+landing&coll=ddd&page=1&facets%5Bspatial%5D%5B%5D=Nederlandse+Antillen&identifier=ddd:010460520:mpeg21:a0167&resultsidentifier=ddd:010460520:mpeg21:a0167&rowid=7)

#### The knowledge base - a vector store
Language models all work with vectors - embedded text. Instead of saving text, a the data has to be stored in embedded versions in a vector store, where the retriever can shop around for the relevant text.

There a number of packages to be used in this section to build the RAG.
```python
import os
from IPython.display import Image, display
from typing_extensions import List, TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_nomic.embeddings import NomicEmbeddings
```

Define the large language model to be used to generate an answer based on provided context:
```python
llm = ChatOllama(model="llama3.1:8b", temperature=0)
```

Define the embeddings model, this is the model to convert our knowledge base texts into vector embeddings and will be used for the retrieval part of the RAG:
```python
embeddings=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
```

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: instructor
For more on Nomic embeddings see: https://python.langchain.com/api_reference/nomic/embeddings/langchain_nomic.embeddings.NomicEmbeddings.html

using inference_model="local" uses (Embed4All)[https://docs.gpt4all.io/old/gpt4all_python_embedding.html]


::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

In the text files, the articles are split by '---'. This information can be used to store the individual articles into a list. Store the filename of the articles in a list as well, so that one can find easily in from which file a text snippet was taken.

```python
dir = "./rag_data"
articles = []
metadata = []

# Iterate over files and add individual articles and corresponding filenames to lists
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    with open(file_path, "r") as f:
        content = f.read().split('---')
        articles.extend(content)
        metadata.extend([file_path] * len(content))
```

The generator will in the end provide an answer based on the text snippet that is retrieved from the knowledge base. If the fragment is very long, it may contain a lot of irrelevant information, which will blur the generated answer. Therefor it is better to split the data into smaller parts, so that the retriever can collect very specific pieces of text to generate an answer from. It is useful to keep some overlap between the splits, so that information does not get lost because of for example splits in the middle of a sentence.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)

documents = text_splitter.create_documents(articles, metadatas=[{'filename': file} for file in files])

print(documents)
```

This text splitter splits text based on the defined character chunk size, but also takes into account spaces and newlines to slit in "smart" chunks, so the chunks will not be exactly of length 500.

Finally, convert each text split into a vector, and save all vectors in a vector store. The text is converted into embeddings using the earlier defined embeddings model.

```python
vectorstore = InMemoryVectorStore.from_texts(
    [doc.page_content for doc in documents],
    embedding=embeddings,
)
```

The contents of the vectorstore can be printed as
```python
print(vectorstore.store)
```
It shows that for each text fragment that was given, a vector is created and it is saved in the vectorstore together with the original text.

#### Setting up the retriever and generator
Define the structure of a dictionary with the keys `question`, `context`, and `answer`.

```python
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
```

Define the retriever function of the RAG. It takes in the question and does a similarity search in the created vectorstore and returns the text snippets that were found to be similar. The similarity search converts the question into an embeddings vector and uses the cosine similarity to determine the similarity between the question and snippets. It then returns the top 4 snippets with the highest cosine similarity score. The snippets are returned in the original text form, i.e. the retrieved vectors are transformed back into text.

```python
def retrieve(state: State):
    "Retrieve documents that are similar to the question."
    retrieved_docs = vectorstore.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}
```

Define the generator function of the RAG. In this function a prompt is defined for the RAG using the context and question. The large language model (the Llama model, defined above) is then invoked with this question and generates an answer for the provided prompt, which is returned as the answer key of the dictionary.

```python
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    rag_prompt = """You are an assistant for question-answering tasks.
    Here is the context to use to answer the question:
    {context}
    Think carefully about the above context.
    Now, review the user question:
    {question}
    Provide an answer to this questions using only the above context.
    Use 10 sentences maximum and keep the answer concise.
    Answer:"""

    rag_prompt_formatted = rag_prompt.format(context=docs_content, question=State["question"])
    
    generate = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"answer": generate.content}
```

#### Build the workflow
The retriever and generator are combined into a workflow graph. The workflow is defined as a StateGraph that uses the dictionary structure (with the keys `question`, `context`, and `answer`) defined above. The retriever and generator are added as nodes, and the two are connected via the edge. The retrieve is set as the start point of the workflow, and finally the graph is compiled into an executable.
```
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge("retrieve", "generate")
workflow.set_entry_point("retrieve")

graph = workflow.compile()
```
```python
display(Image(graph.get_graph().draw_mermaid_png()))https://scikit-learn.org/1.5/modules/grid_search.html
```

![workflow](./fig/workflow_rag.png)

That's it! The RAG can now be asked questions. Let's see what it can tell about the moon landing:

```python
response = graph.invoke({"question": "Who landed on the Moon?"})
print(response["answer"])
```

This is quite a specific answer. It can be seen why by looking at the text snippets that were used:
```
print(response["context"])
```

While a general chatbot uses all the information in the material that it was trained on, the RAG only uses the information that was stored in the vectorstore to generate the answer.

:::::::::::: challenge 
Try generating more answers with the RAG based on other questions, perhaps also looking at the newspaper texts that are used. What stands out?

:::::: solution
For example:
- The RAG returns in some cases that no answer can be generated on the context it was provided with
- For some questions, the LLM returns that it cannot provide an answer because of safety precautions that are inherent to the LLM used, such as information about violent acts.
::::::

::::::::::::

This is the simplest form of a RAG, with a retriever and a generator. However, one can make the RAG more complex by adding more components and options to the workflow, for example one to check the relevance of the retrieved documents, removing those that turn out to be irrelevant to be used for the answer generation, or having a component that can  reformulate the question. Another example is to add a hallucination checker step after the generator that checks if the generated answer can actually be found in the provided context. 

# Pitfalls, limitations, privacy
While LLMs are very powerful and provide us with many great possibilities and opportunities, they also have limitations. 

- Training data: LLMs are trained on large data sets. These are often collected from the internet and books, but these come with downsides 
  - They have a date cutoff; LLMs are trained on a static data set, meaning they are trained on information up to a certain date. They therefor do not have the latest information. They are definitely not useful for recent news events (often they mention this), but also lack behind in for example technological advancements. Always consider how old the model is that you are using.s
  - There is no fact checking involved in the training data. If the training data contains a lot of incorrect information or fake news, this will affect the answers generated.
  - The training data is often biased, including social and cultural biases. This can lead to harmful responses that are stereotyping or racist. Training LLMs does involve a human review for fine-tuning the models, such that they are prevented from answering questions on illegal activities, political advice, advice on harming yourself or others, or generating violent or sexual content. When for example prompted with questions about politics it does provide generic factual information, but will also say that it will not give advice or opinionated answers.
  - For GPT-4, there is no exact information provided as to which data it is trained on, meaning that the data might be breaking privacy laws or copyright infringement.
  - The data an LLM is trained on is generic, resulting in that it is not good at generating answers for specialised questions. There are however already a lot of models that are finetuned for specific fields.
  - Language: LLMs are primarily trained on data collected from the internet, resulting in that they are 'best' in the most spoken languages. ChatGPT is trained on many languages, but languages that are less widely spoken will automatically have smaller data to train on, which makes the LLM less accurate in these languages.
- Multi-step thinking: LLMs are generally not good at multi-step thinking. They are very good at providing bullet point lists of information, but reasoning like humans do, drawing a conclusion from combined logic is something they are not good at (yet).
- Hallucinations: LLMs tend to hallucinate. When it 'does not know the answer', it will often still try to provide an answer. You should therefore not blindly use the answers from an LLM, but still check the given information yourself.
- Privacy: when using a language model locally, such as done above with Llama, your privacy is preserved. The model is only on your laptop, and the data you provide is not uploaded to any server. But when you for example use ChatGPT via the web interface, there is no privacy. Any information you provide, questions, provided context and so on will be used by ChatGPT. It will be used (ao) for improving the model, which may be considered a good thing, but other things it is used for are not necessarily known. This means that you should be careful in what you provide to the LLM. Never provide sensitive information or private data to an LLM that you do not run fully locally.

::::::::::::::::::::::::::::::::::::: keypoints

## Key points to remember


# References

Comprehensive overview of LLMs:
https://dl.acm.org/doi/abs/10.1145/3744746

Andrej Karpathy deep dive into LLMs:
https://www.youtube.com/watch?v=7xTGNNLPyMI

Multilingual benchmark for LLMs:
https://mmluprox.github.io/

:::::::::::::::::::::::::::::::::::::