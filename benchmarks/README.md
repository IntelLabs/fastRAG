# 💻 Benchmarks 💯

We provide scripts for running some well known benchmarks with fastRAG. The goal is to showcase common usecases, such that include BM25 retrievers, Sentence BERT rankers, ColBERT retriever and a FiD generative reader. The scripts can also be used as a starting point to benchmark new and modified workflows with new components.

We report results as compared with the original benchmark papers.

## Installation

Running the benchmarks require some additional packages:

``` sh
pip install beir
pip install git+https://github.com/facebookresearch/KILT.git
```

## :beers: BeIR Benchmark

[BeIR](https://github.com/beir-cellar/beir) contains diverse IR tasks ([Thakur et al. 2021](#org29ef2fc)); here we focus on MSMARCO and NaturalQuestions (NQ). For each task, we test two pipelines:

1. BM25 retriever with a Sentence BERT cross-encoder re-ranker; **scripts:** [MSMARCO](BEIR/msmarco-bm25-sbert.py), [NQ](BEIR/nq-bm25-sbert.py).
2. ColBERT v2 retriever using a PLAID index; **scripts:** [MSMARCO](BEIR/msmarco-plaid.py), [NQ](BEIR/nq-plaid.py).

### Results

|             | nDCG@10 |    20 |    50 |   100 | MAP@10 |    20 |    50 |   100 |  R@10 |    20 |    50 |   100 |  P@10 |     20 |    50 |   100 |
|:------------|--------:|------:|------:|------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|------:|------:|
| **MSMARCO** |         |       |       |       |        |       |       |       |       |       |       |       |       |        |       |       |
| BM25+SBERT  |   .4007 | .4111 | .4159 | .4163 |  .3485 | .3516 | .3524 | .3525 | .5567 | .5962 | .6190 | .6212 | .0582 |  .0312 | .0130 | .0065 |
| PLAID       |   .4610 | .4857 | .5030 | .5103 |  .3889 | .3960 | .3990 | .3997 | .6815 | .7765 | .8608 | .9046 | .0712 | 0.0408 | .0182 | .0096 |
| **NQ**      |         |       |       |       |        |       |       |       |       |       |       |       |       |        |       |       |
| BM25+SBERT  |   .4732 | .4871 | .4971 | .5002 |  .4191 | .4234 | .4253 | .4257 | .6111 | .6631 | .7098 | .7279 | .0725 |  .0396 | .0170 | .0087 |
| PLAID       |   .5411 | .5634 | .5781 | .5828 |  .4700 | .4770 | .4797 | .4802 | .7327 | .8157 | .8843 | .9112 | .0871 |  .0489 | .0213 | .0110 |

## KILT

[KILT](https://github.com/facebookresearch/KILT) contains mutiple Open-Domain tasks, including Question Answering, Slot Filling, Fack Checking, and more ([Petroni et al. 2021](#org29ef2fd)); For evaluation, we primarily showcase its version of NaturalQuestions (NQ). This version differs from the one in BeIR, as it focuses on the downstream performance of the model, not just the retrieval. The pipelines utilized here are essentially the same as the ones in BeIR, with the inclusion of a [FiD](https://github.com/facebookresearch/FiD) model as the reader, which outputs the final answer. The pipelines are:

1. BM25 retriever with a cross encoder SBERT reranker; **scripts:** [NQ](KILT/nq-bm25-fid.py).
2. ColBERT v2 retriever using a PLAID index; **scripts:** [NQ](KILT/nq-plaid-fid.py).

### Results

The following are the performance and efficiency results of the pipelines:

|            | Accuracy | EM    | F1    | Rouge-L |
|:-----------|---------:|------:|------:|--------:|
| **Natural Questions**
| BM25+SBERT | 41.45    | 46.88 | 55.60 | 55.60   |
| PLAID      | 46.14    | 53.29 | 62.31 | 58.78   |

## References

<a id="org29ef2fc"></a>Thakur, Nandan, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.” In *Thirty-Fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*. <https://openreview.net/forum?id=wCu6T5xFjeJ>.


<a id="org29ef2fd"></a>Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and
Sebastian Riedel. 2021. KILT: a benchmark for knowledge intensive language tasks. In *Proceedings
of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies*, pages 2523–2544, Online. Association for Computational Linguistics.
<https://aclanthology.org/2021.naacl-main.200.pdf>.
