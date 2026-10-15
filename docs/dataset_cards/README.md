# Dataset cards

The cards of the two phone-derived datasets on the Hugging Face Hub, kept here like the
[model cards](../model_cards/). `scripts/push_dataset.py` wrote the first version of each; these
add which run used each dataset and drop LeRobot's empty "Paper" and "Citation" placeholders.

| Card | Hub dataset |
|---|---|
| [libero90_frying_pan_phone.md](libero90_frying_pan_phone.md) | [Delcastillo8/libero90_frying_pan_phone](https://huggingface.co/datasets/Delcastillo8/libero90_frying_pan_phone) (153 demos, run Pall) |
| [libero90_frying_pan_phone_50.md](libero90_frying_pan_phone_50.md) | [Delcastillo8/libero90_frying_pan_phone_50](https://huggingface.co/datasets/Delcastillo8/libero90_frying_pan_phone_50) (50 demos, run P) |

To update them on the Hub (logged in with `hf auth login`, from the repository root):

```bash
hf upload Delcastillo8/libero90_frying_pan_phone docs/dataset_cards/libero90_frying_pan_phone.md README.md --repo-type dataset
hf upload Delcastillo8/libero90_frying_pan_phone_50 docs/dataset_cards/libero90_frying_pan_phone_50.md README.md --repo-type dataset
```

Each command replaces only `README.md` in that dataset repository; the data stays as it is.
