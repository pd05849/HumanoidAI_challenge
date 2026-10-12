# Model cards

The model cards of the three fine-tunes. Each file becomes the `README.md` of its Hugging Face
model repository.

| File | Run | Model repository |
|---|---|---|
| [`T.md`](T.md) | T: + 50 LIBERO teleop demos | [`Delcastillo8/smolvla_libero_frying_pan_T`](https://huggingface.co/Delcastillo8/smolvla_libero_frying_pan_T) |
| [`P.md`](P.md) | P: + 50 phone-derived demos | [`Delcastillo8/smolvla_libero_frying_pan_P`](https://huggingface.co/Delcastillo8/smolvla_libero_frying_pan_P) |
| [`Pall.md`](Pall.md) | Pall: + 153 phone-derived demos | [`Delcastillo8/smolvla_libero_frying_pan_Pall`](https://huggingface.co/Delcastillo8/smolvla_libero_frying_pan_Pall) |

## Upload

From the repository root, logged in with a token that can write:

```bash
hf auth login        # once
hf upload Delcastillo8/smolvla_libero_frying_pan_T docs/model_cards/T.md README.md
hf upload Delcastillo8/smolvla_libero_frying_pan_P docs/model_cards/P.md README.md
hf upload Delcastillo8/smolvla_libero_frying_pan_Pall docs/model_cards/Pall.md README.md
```

Each command adds or replaces `README.md` in that repository, in one commit. The model files,
`train.log` and `dataset_build/` stay as they are.

The YAML block at the top of each card is what the Hub reads: license (Apache-2.0, the same as
`smolvla_libero`), `base_model` (the "fine-tuned from" link), the training datasets and the tags.
The results in the cards come from [`results/summary.md`](../../results/summary.md).
