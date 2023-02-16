# Running Demos

To run a demo, use its config name; for example:

```sh
python run_demo.py -t QA1
```

The server and UI are are created as subprocesses that run in the background. Use the PIDs to kill them.

Use the  `--help` flag for a list of available configurations.

## Available Demos

| Name    | Comment                                                                             | Config Name |
|:--------|:------------------------------------------------------------------------------------|:-----------:|
| Q&A     | Abstractive Q&A demo using BM25, SBERT reranker and an FiD.                         | `QA1`       |
| Q&A     | Abstractive Q&A demo using ColBERT v2 (w/ PLAID index) retriever and an FiD reader. | `QA2`       |
| Summary | Summarization using BM25, SBERT reranker and long-T5 reader                         | `SUM`       |
| Image   | Abstractive Q&A demo, with an image generation model for the answer.                | `QADIFF`    |

ColBERT demo with a wikipedia index takes about 15 minutes to load up. Also, see remark about GPU usage in the [README](../README.md#plaid-requirements).
