# Models Overview

## Elasticsearch
<image align="right" src="https://upload.wikimedia.org/wikipedia/commons/f/f4/Elasticsearch_logo.svg" width="360">

[Elasticsearch](https://www.elastic.co/elasticsearch/) is a performant search engine, with an open source license. It's
written in Java, supports JSON documents (noSQL), has an HTTP interface and has clients in many languages. Its full-text
search abilities are based on the Lucene library and mainly uses the [BM25
algorithm](https://en.wikipedia.org/wiki/Okapi_BM25). The trade-off of using elasticsearch: battle-tested DB solution,
very fast querying, good baseline to compare neural-network-based retrievers.

We use the Haystack `ElasticsearchDocumentStore` connector class. It assumes the user already has an elasticsearch
instance running; required arguments include: URL (could be `localhost` if ran locally), index name, port, username and
password.

## ColBERT v2 with PLAID Engine
<image align="right" src="assets/colbert-maxsim.png" width="500">

ColBERT is a dense retriever which means it uses a neural network to encode all the documents into representative
vectors; once a query is made, it encodes the query into a vector and using vector similarity search, it finds the most
relevant documents for that query. What makes it different is the fact it stores the full vector representation of the
documents; neural network represent each word as a vector and previous models used a single vector to represent
documents, no matter how long they were. ColBERT stores the vectors of all the words in all the documents. It makes the
retrieving more accurate, albeit with the price of having a larger index. ColBERT v2 reduces the index size by
compressing the vectors using a technique called quantization. Finally, PLAID improves latency times for ColBERT-based
indexes by using a set of filtering steps, reducing the number of internal candidates to consider, reducing the amount
of computation needed for each query. Overall, ColBERT v2 with PLAID provide state of the art retrieving results with a
much smaller latency than previous dense retrievers, getting very close to sparse retrievers performance with much
higher accuracy. See ([Santhanam, Khattab, Saad-Falcon, et al. 2022](#org330b3f5); [Santhanam, Khattab, Potts, et al. 2022](#orgbfef01e)) for more details.

We provide an implementation of ColBERT and PLAID, exposed using the classes `PLAIDDocumentStore` and
`ColBERTRetriever`, together with a trained model, see [ColBERT-NQ](https://huggingface.co/Intel/ColBERT-NQ). The
document store class requires the following arguments:

- `collection_path` is the path to the documents collection, in the form of a TSV file with columns being
"id,content,title" where the title is optional.
- `checkpoint_path` is the path for the encoder model, needed to encode queries into vectors at run time. Could be a
local path to a model or a model hosted on HuggingFace hub. In order to use our trained **model** based on
NaturalQuestions, provide the path `Intel/ColBERT-NQ`; see [Model
Hub](https://huggingface.co/Intel/ColBERT-NQ) for more details.
- `index_path` location of the indexed documents. The index contains the optimized and compressed vector representation
of all the documents. Index can be created by the user given a collection and a checkpoint, or can be specified via a
path.

**Updated:** new feature that enables adding and removing documents from a given index. Example usage:

```python
index_updater = IndexUpdater(config, searcher, checkpoint)

added_pids = index_updater.add(passages)  # Adding passages
index_updater.remove(pids)                # Removing passages
searcher.search()                         # Search now reflects the added & removed passages

index_updater.persist_to_disk()           # Persist changes to disk
```

---

#### :warning: PLAID Requirements :warning:
>
> If GPU is needed it should be of type RTX 3090 or newer (Ampere) and PyTorch should be installed with CUDA support using:
>
>```bash
>pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
>```


## Fusion-In-Decoder
<image align="right" src="assets/fid.png" width="500">

The Fusion-In-Decoder model (FiD in short) is a transformer-based generative model, that is based on the T5 architecture. For our setting, the model answers a question given a question and relevant information about it. Thus, given a query and a collection of documents, it encodes the question combined with each of the documents simultaneously, and later uses all encoded documents at once to generate each token of the answer at a time. See ([Gautier Izacard, Edouard Grave, et al. 2020](#org330b3f6); ) for more details.

We provide an implementation of FiD, using the class `FiDReader` and
`FusionInDecoderForConditionalGeneration`.

To finetune your own an FiD model, you can use our training script here: [Training FiD](scripts/training/train_fid.py)

The data used for training and evaluation is a .json file (**not** .jsonl), containing json dictionaries as entries. Each entry adheres to the following format:

```json
{
  "id": 0,
  "question": "The question for the current example",
  "target": "Target answer",
  "answers": ["Target answer", "Possible answer #1", "Possible answer #2"],
  "ctxs": [
            {
                "title": "Title of passage #1",
                "text": "Context of passage #1"
            },
            {
                "title": "Title of passage #2",
                "text": "Context of passage #2"
            },
            ...
          ]
}
```

The following is an example command, with the standard parameters for training the FiD model:

```
python scripts/training/train_fid.py
--do_train \
--do_eval \
--output_dir output_dir \
--train_file path/to/train_file \
--validation_file path/to/validation_file \
--passage_count 100 \
--model_name_or_path t5-base \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--seed 42 \
--gradient_accumulation_steps 8 \
--learning_rate 0.00005 \
--optim adamw_hf \
--lr_scheduler_type linear \
--weight_decay 0.01 \
--max_steps 15000 \
--warmup_step 1000 \
--max_seq_length 250 \
--max_answer_length 20 \
--evaluation_strategy steps \
--eval_steps 2500 \
--eval_accumulation_steps 1 \
--gradient_checkpointing \
--bf16 \
--bf16_full_eval
```

## References

<a id="orgbfef01e"></a>Santhanam, Keshav, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. “PLAID: An Efficient Engine for Late Interaction Retrieval.” arXiv. <https://doi.org/10.48550/arXiv.2205.09707>.

<a id="org330b3f5"></a>Santhanam, Keshav, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. “ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.” arXiv. <https://doi.org/10.48550/arXiv.2112.01488>.

<a id="org330b3f6"></a>Gautier Izacard and Edouard Grave. 2020b. Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint
arXiv:2007.01282.
