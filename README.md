### Dialogue generation with Transformers

This project tries to generate dialogue utterances with an Encoder-Decoder built with a [transformer](http://jalammar.github.io/illustrated-transformer/) architecture. The dataset used for it are dialogues from [Opensubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php).


#### The dataset

The dataset used for the project can be downloaded [here](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.en.gz), and it contains 140 million utterances approximately. More dialogue datasets can be found in [this survey](https://breakend.github.io/DialogDatasets/).

The opensubtitles dataset is pretty dirty, with some metadata of the movies, comments not being part of the dialogues, descriptions, ...

I have used two different sets of utterances:
* small: 80.000 utterances approximately. These are the json files found inside the json_small folder.
* big: 2M utterances approximately. These are the json files found inside the json_big folder. 

We have used some [scripts](https://github.com/PolyAI-LDN/conversational-datasets/tree/master/opensubtitles) from Poly AI to clean and format the dataset. The script generates an output containing each of the utterances as shown below:


```
[Context]:
	Oh, my God, we killed her.
[Response]:
	Artificial intelligences cannot, by definition, be killed, Dr. Palmer.

Extra Contexts:
	[context/9]:
		So what are we waiting for?
	[context/8]:
		Nothing, it...
	[context/7]:
		It's just if...
	[context/6]:
		If we've underestimated the size of the artifact's data stream...
	[context/5]:
		We'll fry the ship's CPU and we'll all spend the rest of our lives stranded in the Temporal Zone.
	[context/4]:
		The ship's CPU has a name.
	[context/3]:
		Sorry, Gideon.
	[context/2]:
		Can we at least talk about this before you connect...
	[context/1]:
		Gideon?
	[context/0]:
		You still there?

Other features:
	[file_id]:
		lines-emk
```

Where the *context* are the previous dialogue sentences to the *response*. It is highly configurable and in our case we have generated it as :

* Just one context sentence. We just use the previous sentence and there is a huge performance gain when creating the files. 
* The script can generate the output as serialized Tensorflow format or JSON. We have created JSON outputs. 
* It also manages the dataset split. We created just one file with all the dialogues.

We have done some small modifications to the origial script, and this can be found in the respsitory (create_data.py)

Although we have transformed the whole dataset we end up using just a small part of it.

#### The model

The implementation of the model used (found at [model.py](https://github.com/uabinf/nlp-fall-2019-project-transformer-dialogue/blob/master/model.py)) is from the [tensorflow implementation](https://www.tensorflow.org/tutorials/text/transformer) of the translator having cleaned it and changed it to a script mode.

#### Files and folders

* checkpoints_big: folder with the checkpoints for the trained model with the recommended setup from the paper 'Attention is all you need' and the big dataset
* checkpoints_small: folder with the checkpoints for the trained model with the small architecture and the small dataset
* json_big: folder containing one file with the big dataset already transformed to json by the PolyAI script
* json_small: folder containing a set of files with the small dataset. This is splitted into different files as this is how the PolyAI script works by default. The test files are not used
* clean_data.py: script executing a bash command for the create_data.py and its options. I had to do it this way in order to avoid having to modify all the script
* create_data.py: modified version of the PolyAI script to create the JSON files from the Opensubtitles dataset. I did a small modification to solve one problem when encoding one of the files name and also modifying some of the default parameters values as it didn't seem to work when calling them from the command line
* evaluate.py: not a trully 'evaluation'. What it does is give the response to a given sentence. It first creates the dictionary of tokens from the dataset, loads the checkpoint and generates the response. See below how to change from the small dataset to the big one
* model.py: the model parts implementation from [tensorflow](https://www.tensorflow.org/tutorials/text/transformer)
* test.json: sample file of the output from the PolyAI script
* tokenizer.py: script to generate the dictionary from the json records used for training and evaluation 
* train.py: the actual model training

#### Evaluation
To evaluate the model we just generate responses to context sentences. I have done this with different datasets and architectures, two of them are available in the repository. For each one there is:
* The checkpoint
* The JSON files

To see the results with the already trained models just run the evaluate.py script.

To switch from one to the other set the path variables inside the tokenizer.py and the evaluate.py to point to the correct paths. The checkpoints are generated with the corresponding dataset so to have the proper results set both of them to either on or the other

#### Poster

The poster latex code and the generated pdf can be found inside the poster folder

#### Some resources that helped me to understand self-attention and transformers:

Videos:
* [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 8 – Translation, Seq2Seq, Attention](https://www.youtube.com/watch?v=XXtpJxZBa2c&t=3905s)
* [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 14 – Transformers and Self-Attention](https://www.youtube.com/watch?v=5vcj8kSwBCY&t=322s)
* [Transformer Attention is all you need](https://www.youtube.com/watch?v=z1xs9jdZnuY&t=635s)

Papers, presentations and blog entries:
* [The ilustrated transformer](http://jalammar.github.io/illustrated-transformer/)
* [Standford NLP slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture14-transformers.pdf), from the previous videos
* And obbiously [Attention is all you need](https://arxiv.org/abs/1706.03762)

