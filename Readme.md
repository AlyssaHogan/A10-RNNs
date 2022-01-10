# Assignment 10: Recurrent Neural Networks (RNN's)

Adapted by Mark Sherman <shermanm@emmanuel.edu> from MIT 6.S198 under Creative Commons
Emmanuel College - IDDS 2132 Practical Machine Learning - Spring 2021

This assignment is based on work by Kevin Zhang, Harini Suresh, Wendy Wei, Martin Schneider, Natalie Lao, and Hal Abelson

0\. Introduction to RNN's (In-class Monday)

Until now, our neural networks have been been single use - once trained, we gave them an input, and they returned an output, but remembered nothing about previous ones. Often times, we would like our networks to make decisions that rely on some memory of what came before: For example, when processing a video clip, each frame should be interpreted by the network in the context of previously seen frames.

In this unit, we will learn about a new type of neural network architecture that has memory, and that allows us to effectively model sequenced data. Recurrent neural networks have been successfully used for natural language processing tasks such as sentiment analysis and machine translation, and they can be adapted to work for many other machine learning tasks like music and image generation.

We will focus on modeling sentences in this assignment. The diagram in Figure 1 represents the data flow through a single RNN cell. In order to model a sequence of words, we define input x_t to be a vector representing the word at time t. The network combines this input with s_t, the hidden state at time t, to produce a new hidden state for the next cell at time t+1. This state represents its "memory" of previous inputs. It generally does not contain the exact values of previous inputs, but rather a different representation that summarizes meaningful features (for example, the cell may care about whether the previous word was a verb or a noun, but not how many letters it had).

Given a series of inputs, we apply our RNN cell sequentially to each input, updating the hidden state as we go along. Figure 2 shows a simple RNN with outputs y_t at each time step. In the figure that each circle is the same cell, but at a different time step, with s_t being the value recurring (memorized) from the previous time step.

Single Cell

![](https://lh5.googleusercontent.com/Gm50iMd3z7FdzegCnxorYEp6dhrnd60JxM6iHeqsOZvecOuZRqHk6a1qFrPPbBjVTGrFJ_c2P3NqRLbqN6b6Oogpn4t9649HQ6v-sssvKu1ineE6TyHCXYHDNfEUt_0Z=w360)

0.1 Simple RNN

We will examine two concrete implementations of recurrent network cells. The original recurrent neural network (RNN) proposed by Jeff Elman in 1990 has largely been superseded by the long-short term memory (LSTM) and gated recurrent units (GRU), but it can still be used as a simple baseline.

Figure 3 shows the equations for our simple RNN cell. The new hidden state is a function of the previous hidden state and the current input. By applying this recursively, we see that the hidden state is a function of all previous inputs.

In practice, however, simple RNNs often fail to learn long-term dependencies, which is why researchers have proposed alternatives such as the LSTM and GRU.

![](https://lh4.googleusercontent.com/fpBcFtjBaNwhQQtIZqsYxyOTH91zYEj51wfVNERYj69GXwLgJG7OoSoQM9fbB-bxv5Ln418oqWHQsBYmOaHuRRQFVH9IZT1p8ADy-XcqmC_hzs0uS-EJmvshfoda9d0Q=w470)

Equations for simple RNN

![](https://lh3.googleusercontent.com/EHaH1s-OYvkWEpp3wSbWBApF2S-x5O54xCd8kDPehj_UIVxh6sRGwp3RnRumrU98Wv_rzcmWf9Q7Jq5mJbKre2dmP9_HMyzyk02QRsVsTAjHXD_i5taAGn0GiEp6dli-=w332)

0.2 LSTM

The original equations for the long-short term memory (LSTM) network are presented below. The f, i, and o variables are referred to as the forget, input, and output gates. They allow the model to explicitly control how much of the hidden state to forget, how much of the hidden state to replace with new inputs, and how much of the hidden state to show the "world".

By explicitly allowing our model to control what to remember and what to forget, we see much better performance on tasks that require long-term memory.

We will take a deeper look at a really good (aka much better) explanation of what is going on with LSTMs (with pretty pictures too).

Equations for original LSTM

![](https://lh4.googleusercontent.com/xloQjARU8ZjKjJSH2OxhcghCMcs3RCQFrJHPt1S3VleCJN-c5SQxnb7CtoKEDVOo--e1MylSay9T5tHZtJTaGcH3H-bCLk4DeYaiXVwtQp2k4NloC5vKcGZbCBky-us3=w446)

Recurrent neural networks are very powerful... but they are also difficult to train.

Due to their sequential nature, the full sequence of outputs can be computed only after the entire sequence of inputs has been processed, which means training cannot be fully parallelized. Furthermore, exploding and vanishing gradients are a much bigger problem in recurrent networks than in feedforward or convolutional networks.

Fortunately in this assignment, we will train small networks. You may still need to wait 10~15 minutes for training to complete.

# 1: Text Generation

Text classification was an example of a "many-to-one" type of RNN, where many inputs (words in a sentence) are turned into one output (the classification label). Now we introduce a "many-to-many" RNN, which we will call a sequence-to-sequence RNN. Commonly used in machine translation, sequence-to-sequence RNN's can be used to generate text. Given a sequence of words, an RNN can predict the probability of the next word, and so intuitively, a generative RNN chooses the most likely next word again and again as it builds a sentence.

To get intuition on how the training works, play around with this interactive demo by Andrej Karpathy in your browser: <https://cs.stanford.edu/people/karpathy/recurrentjs/>. This demo generates new sentences character by character, using user-provided data from the input text box. To use the demo, press learn/restart in the controls section and look at the "sentences" generated in the Model Samples section. You can use Pause and Restart to make the output easier to follow. Note that the modeling happens character by character: any actual words produced are because the character sequence was learned, and there are lots of strings that are not words. The button at the bottom lets you load a pretrained model so you can see what output would look like after 100 hours of training.

For homework, we will experiment with generating sentences using at least two input datasets to train a single model. Our goal is to generate texts that combine the different styles of our input datasets. One amusing example of this is "King James Programming", trained using input from the King James Bible mixed with Abelson and Sussman's "Structure and Interpretation of Computer Programs": <http://kingjamesprogramming.tumblr.com/>. (The output shown here was generated with a Markov chains rather than an RNN.)

An example of what King James Programming bot saith:

"commutativity of addition is a single theorem because it depends on the kind of unholy rapport he felt to exist between his mind and that lurking horror in the distant black valley."

"hath it not been for the singular taste of old Unix, "new Unix" would not exist."

## 1.1 Setup

We will use a modified version of the Karpathy's demo code, written in RecurrentJS, which is a Javascript library (different from Deeplearn.js) that implements RNNs and LSTMs. 

* Clone this assignment. 
* Open the assignment folder in Visual Studio Code.
* Use the Live Server plugin to "Go Live" - which will open a browser window.
* In the browser, click `character_demo.html`

The demo will come up when you open the page. As usual also open the browser developer tools Javascript console to help you see what is happening.

Note: Slightly different from the previous assignments this semester, the Javascript code here is embedded in the html file rather than loaded as a separate file. To make changes, just edit the html file and reload the page. 

*Pro Tip:* the Live Server plugin for VSC automatically reloads whenever you save the file! One less step!

To get code-checking (which I **highly** recommend):

* Open the assignment folder in the terminal, and run `npm install`
* Use the ESLint plugin for VSC (I marked it as recommended for you)

## 1.2 Complete the Code

If you load the code as is, you will get errors. To fix the errors, you'll need to complete the code at line 273 marked TODO.\
This requires uncommenting the two declarations of the variable out_struct (one for RNNs and one for LSTMs) and filling in the missing arguments to `R.forwardRNN` `R.forwardLSTM`, the functions that perform the forward propagation through one cell of the RNN or LSTM

Hint: The 5 arguments to be filled in are the same in both places. They are (in order):
- the graph G
- the model
- [hidden_sizes], the list of sizes of the hidden layers
- the input row to the cell, here denoted x
- the previous cell, here denoted prev

If you fill in these inputs and refresh the page, things should load without error: The LSTM model should start to run and you should start seeing output generated toward the bottom of the page.

Note: Documentation on RecurrentJS is here: <https://github.com/karpathy/recurrentjs>

## 1.3 Run the model

### Problem 1 - WRITEUP REQUIRED
1\. Run the model for 10 to 15 epochs, or until you see interesting results. Pause the model and record the perplexity. Perplexity is a measurement of how well the model predicts a sample. A low perplexity indicates that the model is good at making predictions.

> What is the perplexity? what does this value indicate to you?

### Problem 2 - WRITEUP REQUIRED
2\. Softmax sample temperature is a hyperparameter that determines how softmax computes the log probabilities of the prediction outputs. If the temperature is high, the probabilities will go toward zero and you will see less frequent words. If the temperature is low, then you will see more common words, but there may be more repetition.

Try three different ranges of softmax temperature: one between `0.2 - 0.3`, one between `0.3 - 0.5` and one between `0.5 - 1.0`. Choose a value in each range and note the results for each value. 

Choose one of the three ranges to fine-tune, which you will use in the following problems. Based on your chosen range, try to find a temperature that produces the most natural seeming text, and give some examples of your generated sentence results.

> What temperature did you find? Show examples of results. 

### Problem 3 - WRITEUP REQUIRED
3\. Write down any observations about your generated sentence results. Does your text reflect properties of the input sources you used (i.e. vocabulary, sentence length)?

> observations

### Problem 4 - WRITEUP REQUIRED
4\. Try changing the model parameters and initialization. Record your observations from at least one of these experiments. Some ideas are:
- Increase or decrease the embedding size for the inputs
- Increase or decrease the size of the hidden layers of each cell
- Adjust the learning rate (Be careful: if the learning rate is too high, the perplexity may explode.)
- Change the generator from LSTM to RNN.

> Document what you tried, what you expected, and what you observed. Include any other notes of relevance.

### Problem 5* - Run with new data (Optional)

The code is preloaded with data. But you can download data of your choice to run this with, such as from <https://www.kaggle.com/paultimothymooney/poetry/data>.

Just make sure each "sentence" is on a different line.

> What new database did you load? Take a few screenshots of some sentences you were able to create.

# Submission
Commit and push your changes to this repository.
