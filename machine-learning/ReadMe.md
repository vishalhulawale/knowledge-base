## Machine Learning
Discipline within Artificial Intelligence that teaches computers how to make predictions or decisions based on data.

![alt text](machine-learning.drawio.png)

[Machine Learning Playbook](../../../Downloads/Playbook-Executive-Briefing-Artificial-Intelligence.pdf)

# Supervised Learning
- Type of machine learning where the model is trained on labeled data.
- The model learns to map inputs to outputs based on the provided labels.
- Y = f(X) + ε where: Y is the output, X is the input, f is the function learned by the model, and ε is the error term.
- Input is called "features" and output is called "label".
    ## Regression
    - Branch of machine learning focused on predicting continuous outcomes.

    ## Classification
    - Branch of machine learning focused on predicting categorical outcomes.
    - Binary Classification - Classifying data into two distinct categories.
    - Multi-class Classification - Classifying data into more than two categories.
    - Support Vector Machines (SVM) - A supervised learning model that finds the hyperplane that best separates different classes in the feature space.
    ![alt text](image.png)

# Unsupervised Learning
- Type of machine learning where the model is trained on unlabeled data.
    ## Clustering
    - Branch of machine learning focused on grouping similar data points together without predefined labels.
    - Use cases include customer segmentation, anomaly detection, and image compression.

    ## Dimensionality Reduction
    - Branch of machine learning focused on reducing the number of features in a dataset while preserving its essential structure.
    - In supervised l;earning, one BIG challenge is to handle the number of features that the algorithm has to deal with.

# Reinforcement Learning
- Type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize a reward signal.
- Perform complex objectives while performing multiple sequence of actions.

## Association Rule Learning
- Branch of machine learning focused on discovering interesting relationships between variables in large datasets.

## Deep Learning
- Subset of machine learning that uses neural networks with many layers to analyze various factors of data.
- Complex Algorithms
- More computing resources required

## Shallow Learning
- Refers to traditional machine learning techniques that do not involve deep neural networks.

## Artificial Neural Networks (ANN)
![alt text](image-1.png)

- Input data is divided into features.
- Each feature is assigned a weight.
- The weighted sum of the features is passed through an activation function to produce an output.
- The model is trained by adjusting the weights based on the error between the predicted output and the actual output.

## Deep Learning Architectures
### Recurrent Neural Networks (RNN)
- Type of neural network designed to handle sequential data.
- Processes input data sequentially, meaning it takes into account the order of the data points. Slowly adjusting the internal state based on the input sequence.

### Convolutional Neural Networks (CNN)
- Type of neural network designed to process grid-like data, such as images.

### Transformer
- Process input data in parallel rather than sequentially.
- Leverage GPUs for faster training and inference.
- Attention Mechanism - Allows the model to focus on different parts of the input sequence when making predictions.

## Foundation Models
- Large-scale pre-trained models that can be fine-tuned for specific tasks.
- Trained on massive datasets and can be adapted to various applications.
- Can be adapted to different tasks with minimal additional training.
- One of most popular foundation models is Generative Pre-trained Transformer (GPT).

## Large Language Models (LLM)
- A type of foundation model specifically designed for natural language processing tasks.
- Handle text input and output, making them suitable for tasks like text generation, translation, and summarization.
- ChatGPT is based on LLM.
- Not all LLms are equal. Model size or number of parameters is a key factor in determining the model's capabilities.
![alt text](image-2.png)
- Bigger model isn't always better. It depends on the task and the data.
- It predicts next token in a sequence based on the context provided by previous tokens.
- Probability distribution over the vocabulary is generated for each token in the sequence.

## LLM Model Types
1.  General Purpose LLMs
    - Designed to handle a wide range of tasks and domains.
    - Trained by taking massive amounts of text data from the internet.
    - ChatGPT, Gemini, Claude, and Llama are examples of general-purpose LLMs.
2.  Domain-Specific LLMs
    - Tailored for specific tasks or industries, such as legal, medical, or technical domains.
    - Fine-tuned on specialized datasets to improve performance in their respective areas.

## Tokens
- Tokens are the basic units of text that LLMs process.
- They can be words, subwords, or characters, depending on the tokenization method used.
- Tokenizer is a tool that converts text into tokens and vice versa.
https://platform.openai.com/tokenizer

## Context Window
- The context window is the maximum number of tokens that an LLM can process at once.
- It determines how much information the model can consider when generating responses.
![alt text](image-3.png)