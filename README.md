# VanillaBertForPassageRetrieval

This is a vanilla Bert for passage retrieval task which takes a query as input and output relevance score for each document in database.

## Data

The whole structure is designed for [CLIRMatrix](https://aclanthology.org/2020.emnlp-main.340/) dataset. However, it is easy to adapt to other datasets by converting your preferred data into the form below:

```json
{
    "src_id": your query id,
    "src_query": query text,
    "tgt_results": [[passage_id, relevance score]]
}
```

For example,

```json
{
    "src_id": "39",
    "src_query": "Albedo",
    "tgt_results": [["553658", 6], ["1712206", 3], ["1849020", 1], ["1841381", 0]]
}
```

## Training

To train, put your data under `data/`. Specify your training, validation and test data path, your document path (target language), and where to save your result.

You can start training by basic commands:

```bash
python train.py \
    --save_dir ckpts/en2zh \
    --target_language zh \
    --train_datapath data/en2zh/en.zh.train.base.jl \
    --val_datapath data/en2zh/en.zh.dev.base.jl \
    --test_datapath data/en2zh/en.zh.test1.base.jl
```

You can find other parameter settings in `utils/parser.py`.

## Inference

To test, specify where your checkpoint is and your test data path.

You can start inference by the following commands:

```bash
python test.py \
    --resume_path ckpts/en2zh/best.pt \
    --target_language zh \
    --test_datapath data/en2zh/en.zh.test1.base.jl
```

It will calculate NDCG@1, NDCG@5 and NDCG@10 for you.

Also, it has codes that allow you to perform inference over all documents in your data. You can check them in `trainer/trainer.py` and `utils/preprocessor.py`. Warning: I haven't improved its efficiency, so it will cost huge amounts of time, but feel free to try.
