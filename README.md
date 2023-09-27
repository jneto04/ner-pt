# Assessing the Impact of Contextual Embeddings for Portuguese Named Entity Recognition

Modern approaches to Named Entity Recognition (NER) use neural networks (NN) to automatically extract features from text and seamlessly integrate them with sequence taggers in an end-to-end fashion.
Word embeddings, which are a side product of pretrained neural language models (LMs), are key ingredients to boost the performance of NER systems.
More recently, contextual word embeddings, which adapt according to the context where the word appears, have proved to be an invaluable resource to improve NER systems.
In this work, we assess how different combinations of (shallow) word embeddings and contextual embeddings impact NER for the Portuguese Language.
We show a comparative study of 16 different combinations of shallow and contextual embeddings and explore how textual diversity and the size of training corpora used in LMs impact our NER results.
We evaluate NER performance using the HAREM corpus.
Our best NER system outperforms the state-of-the-art in Portuguese NER by 5.99 in absolute percentage points. State-of-The-Art results evaluated by CoNLL-2002 Script.


Results for the Total Scenario (HAREM)

| Approach   |    Precision   | Recall |   F1   |
|:----------:|:--------------:|:------:|:------:|
| BiLSTM-CRF+FlairBBP |     **74.91%**     | **74.37%** | **74.64%** |
| BiLSTM-CRF [(Castro, et al.)](https://www.researchgate.net/publication/326301193_Portuguese_Named_Entity_Recognition_using_LSTM-CRF) |     72.28%     | 68.03% | 70.33% |
|   CharWNN [(dos Santos, et al.)](https://arxiv.org/pdf/1505.05008.pdf)  |     67.16%     | 63.74% | 65.41% |

Results for the Selective Scenario (HAREM)

| Approach   |    Precision   | Recall |   F1   |
|:----------:|:--------------:|:------:|:------:|
| BiLSTM-CRF+FlairBBP|       **83.38%**       | **81.17%** | **82.26%** |
| BiLSTM-CRF [(Castro, et al.)](https://www.researchgate.net/publication/326301193_Portuguese_Named_Entity_Recognition_using_LSTM-CRF) |       78.26%       | 74.39% | 76.27% |
|   CharWNN [(dos Santos, et al.)](https://arxiv.org/pdf/1505.05008.pdf) |       73.98%       | 68.68% |    65.41%    |

## Reproduce our tests for NER

Before you begin, you should download the Flair library. Flair is a powerful NLP library with state-of-the-art results. Flair was developed by [Zalando Research](https://research.zalando.com/). You can see all details in this github [link](https://github.com/zalandoresearch/flair).
* Paper: Contextual String Embeddings for Sequence Labeling [(Akbik, et al.)](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view)

STEP 1: Download our language model FlairBBP (backward and forward);

STEP 2: Clone this repository; 

STEP 3: Install Flair. See how to install [here](https://github.com/zalandoresearch/flair);

STEP 4: Download NILC's Word Embedding. You must download Word2Vec-Skip-Gram with 300 dimensions; Put the file inside the cloned folder;

STEP 5: Run our script ```python3.6 ner_flair.py```

## Tagging your portuguese text with our NER model

Tag your text using our best model for NER. The model is formed by FlairBBP + NILC-Word2Vec-Skpg-300d. It is possible to recognize the following categories: PERSON, LOCATION, ORGANIZATION, TIME and VALUE. You need install the last version of Flair.

STEP 1: Download our NER model [Download Here!](https://drive.google.com/file/d/1DirvI87wPS_l3G7AehbvGbZv0X8rCBVP/view?usp=sharing);

STEP 2: Use the [_pToolNER_](https://github.com/jneto04/pToolNER) to labelling your text.

```python
pToolNER = PortugueseToolNER()

pToolNER.loadNamedEntityModel('best-model.pt')

pToolNER.sequenceTaggingOnText(
               rootFolderPath='./PredictablesFiles',
               fileExtension='.txt',
               useTokenizer=True,
               maskNamedEntity=False,
               createOutputFile=True,
               outputFilePath='./TaggedTexts',
               outputFormat='plain',
               createOutputListSpans=True
               )
```

**Alternative use _(We strongly recommend you to use the pToolNER!)_:**

STEP 1: Download our NER model [Download Here!](https://drive.google.com/file/d/1DirvI87wPS_l3G7AehbvGbZv0X8rCBVP/view?usp=sharing);

STEP 2: Clone this repository;

STEP 3: Run our script ```python3.6 tagging_ner.py [input_file_name.txt] [output_file_name.txt] [mode]``` modes:
* conll - input text in conll formart
* plain - input text in plain formart

## Language Models

### Flair Embeddings - FlairBBP
You can download our Flair Embeddings models (FlairBBP) in the following links:
* **Backward:** [FlairBBP-Backward](https://drive.google.com/file/d/1skW89mlm94-fk9NhV-0M7cX4ZRKpDydH/view?usp=sharing)
* **Forward:** [FlairBBP-Forward](https://drive.google.com/file/d/1dqqce73PHZbU70ITalt64zz6lGHvbb6c/view?usp=sharing)

### Word Embeddings
You can download our Word Embedding models in the following links, note that all models were trained in 300 dimensions:


| Algorithm  | Architecture | Downloads |
| ------------- | ------------- | ------------- |
| Word2Vec  | Skip-Gram  | [Word2Vec_skpg_300d](https://drive.google.com/file/d/1GC1iS8hiobBF94UJY5r2vT5caGlvPfoD/view?usp=sharing) |
| Word2Vec  | CBOW  | [Word2Vec_cbow_300d](https://drive.google.com/file/d/1-3NO_ladhqkFG3Y7f_-5PoirymUCwe1k/view?usp=sharing) |
| FastText   | Skip-Gram  | [Fasttext_skpg_300d](https://drive.google.com/file/d/1K8Sg4AA1Nnr0beiBhMRiXFE0yQQjRVV8/view?usp=sharing) |
| FastText   | CBOW  | [Fasttext_cbow_300d](https://drive.google.com/file/d/18R-DF0MevrbVH4lP7t6IQi2a9xfEO3uX/view?usp=sharing) |

### NILC Word Embeddings
You can download the Word Embeddings provided by NILC in the following link: http://nilc.icmc.usp.br/embeddings
* Paper: Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks [(Hartmann, et al.)](https://arxiv.org/pdf/1708.06025.pdf)

## Language Models Corpora

### BlogSet-BR
BlogSet-BR is a large corpus built from millions of sentences taken from Brazilian Portuguese web blogs.
* Paper: BlogSet-BR: A Brazilian Portuguese Blog Corpus [(Santos, et al.)](http://www.lrec-conf.org/proceedings/lrec2018/summaries/10.html)
* [Download Here!](http://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/blogset-br/)

### brWaC
brWaC is another portuguese large corpus.
* Paper: The brWaC Corpus: A New Open Resource for Brazilian Portuguese [(Filho, et al.)](https://www.researchgate.net/publication/326303825_The_brWaC_Corpus_A_New_Open_Resource_for_Brazilian_Portuguese)
* [Download Here!](http://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC)

### ptwiki-20190301
ptwiki-20190301 is a corpus formed by texts from wikipedia in Portuguese.
* [Download Here!](https://dumps.wikimedia.org/ptwiki/20190301/)

Language Model Corpora Size Details (after pre-processing):

|      Corpus     |  Sentences  |     Tokens    |
|:---------------:|:-----------:|:-------------:|
|      brWaC      | 127,272,109 | 2,930,573,938 |
|    BlogSet-BR   |  58,494,090 | 1,807,669,068 |
| ptwiki-20190301 |  7,053,954  |  162,109,057  |
|   All Corpora   | 192,820,153 | 4,900,352,063 |

# Citing our Paper
```
@inproceedings{santos2019assessing,
  author    = {Joaquim Santos and
               Bernardo Consoli and
               Cicero dos Santos and
               Juliano Terra and
               Sandra Collonini and
               Renata Vieira},
  title     = {Assessing the Impact of Contextual Embeddings for Portuguese Named Entity Recognition},
  booktitle = {Proceedings of the 8th Brazilian Conference on Intelligent Systems},
  pages     = {437--442},
  year      = {2019}
}
```
