# SigLIP Baseline

This baseline uses the local frozen SigLIP checkpoint as a strong vision-language
feature extractor:

```text
/mnt/e/project/SceneReVis/ckpt/google/siglip-so400m-patch14-384
```

It extracts one feature vector per AI2-THOR oracle record:

```text
[image_emb, text_emb, image_emb * text_emb, abs(image_emb - text_emb)]
```

The downstream policy is the same gain-score policy used by the Qwen3-VL model:
depth CNN + pose MLP + gated fusion + gain-score regression/ranking loss.

## Dependency

SigLIP tokenization requires SentencePiece:

```bash
pip install sentencepiece
```

If pip is not usable, install through conda:

```bash
conda install -c conda-forge sentencepiece -y
```

## Run

Extract frozen SigLIP features:

```bash
bash scripts/extract_siglip_features.sh --batch-size 4
```

If GPU memory is tight, use:

```bash
bash scripts/extract_siglip_features.sh --batch-size 1
```

Train the gain-score policy:

```bash
bash scripts/train_ai2thor_siglip_policy.sh configs/train_ai2thor_siglip.yaml
```

Evaluate with the same Stay threshold used by the Qwen3-VL model:

```bash
bash scripts/evaluate_ai2thor_siglip_policy.sh --stay-threshold 200
```

Compare Qwen3-VL, SigLIP, and non-learned baselines:

```bash
bash scripts/evaluate_learned_baseline_suite.sh
```
