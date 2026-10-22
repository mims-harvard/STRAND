## PModule

Project-specific modules.

### `DNA_modeling.py`

Contains `BorzoiPerturbAdapter`, which wraps the Borzoi/Flashzoi backbone from `borzoi-pytorch` and produces a single perturbation embedding per sample.

External citation:
- `borzoi-pytorch` / Flashzoi: `https://github.com/johahi/borzoi-pytorch`

Notes:
- `flash-attn` is optional but strongly recommended for Flashzoi speed.
- `peft` is optional (only needed if you enable LoRA fine-tuning components).

