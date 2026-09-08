# Phase 0 · Task 2 — Image Classification

**2026-09-08** · fast.ai Lesson 1 ("Is it a bird?") · run on Kaggle (GPU T4)

Built an image classifier that tells birds from forests, going through the whole pipeline
myself — collecting the data, cleaning it, labelling it, fine-tuning a pretrained model,
and predicting on an image the model had never seen.

The point was not the classifier. It was seeing where every step sits.

---

## The pipeline

```
1  COLLECT      scrape bird / forest images
2  CLEAN        delete corrupted files
3  LABEL        parent folder name becomes the label
4  PREPROCESS   resize everything to 192×192
5  SPLIT        80% train / 20% validation
6  DATALOADER   feed the model 32 images at a time
7  MODEL        pretrained ResNet18
8  FINE-TUNE    3 epochs
9  EVALUATE     error_rate
10 PREDICT      classify an unseen image
```

---

## What each step was actually for

### Data comes before the model

```python
def search_images(keywords, max_images=200): ...
search_images("bird photos")
search_images("forest photos")
```

A model with no data learns nothing. Getting the data is part of the work, not a
prerequisite to it.

### Cleaning

```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
```

Scraped images include broken files. In real ML, improving the data usually beats
improving the model.

### X and y

```
X = input   →  the image
y = label   →  bird / forest
```

```python
blocks=(ImageBlock, CategoryBlock)   # X is an image, y is a category
get_y=parent_label                   # the folder name is the answer
```

```
dataset/
├── bird/     → every file in here is labelled "bird"
└── forest/   → every file in here is labelled "forest"
```

### Train / validation split

```python
RandomSplitter(valid_pct=0.2, seed=42)
```

If a model sees all the data during training, it can memorise it and still score 100%.
That number means nothing. Holding back 20% answers a different question:

> **Can it get images right that it never studied?**

That is the only number worth reading.

### Preprocessing

```python
Resize(192, method='squish')
```

Scraped images arrive at every size. The model needs one fixed shape.

### Batches

```python
.dataloaders(path, bs=32)
```

Images go in 32 at a time rather than all at once.

### Not training from scratch

`resnet18` has already been trained on a very large image dataset. It arrives already
knowing how to see — edges, textures, shapes, parts. That ability is not bird-specific;
it transfers to almost any natural image.

```
pretrained ResNet18  →  fine-tune on bird/forest  →  bird/forest classifier
```

So the training here is not teaching the model to see. It is teaching it what to call
what it already sees.

### Fine-tuning

```python
learn.fine_tune(3)
```

Two tables print, because this happens in two stages:

1. Freeze the pretrained layers, train only the new bird/forest head.
2. Unfreeze and adjust the whole network with a very small learning rate.

Stage 1 comes first so that a few hundred images don't destroy what a million images
already taught the network.

### Epochs and error rate

One epoch = one full pass over the training data. `fine_tune(3)` runs three.

```
error_rate = 0.01   →  wrong 1% of the time  →  99% accuracy
```

### Prediction

```python
is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
```

```
bird     0.998
forest   0.002
```

---

## Terms

| Term | Working definition |
|---|---|
| Dataset | All the data used |
| X | The input — here, the image |
| y / label | The answer — here, bird or forest |
| Training data | What the model studies |
| Validation data | Held back, used to check whether it actually learned |
| Data cleaning | Removing broken or useless data |
| Preprocessing | Reshaping data so the model can take it |
| Batch | How many samples go in at once |
| Epoch | One full pass over the training data |
| ResNet18 | The image-recognition network used here |
| Pretrained model | Already trained on a large dataset |
| Fine-tuning | Further training a pretrained model for your own task |
| Error rate | Fraction of predictions that were wrong |
| Prediction | Running the trained model on new data |

---

## The shape worth keeping

> **Get data → build X and y → split train/validation → fine-tune a pretrained model
> → measure it on data it never saw.**

Dogs vs cats, car models, product categories, medical images — the code changes, this
shape does not.

---

## What went wrong

Most of the time today went to environment problems, not to machine learning.

**Kaggle floods the output with warnings.** `jupyter_client` emits a `DeprecationWarning`
on every message it passes, so anything with a progress bar produces thousands of red
lines. I read that as a crash and interrupted a cell that was running fine.

> Red output is not an error. The test is whether there is a `Traceback` and whether
> the last line reads `SomethingError:`. Nothing else counts.

**The tutorial's dependency had been renamed.** `duckduckgo_search` is now `ddgs`.
The lesson was recorded in 2022; the libraries around it have moved.

**The most fragile part was the part that touched the outside world.** Scraping images
from a search engine broke; nothing about the model did. Worth remembering when choosing
what a project depends on.

**Fix for next time:** put `import warnings; warnings.simplefilter('ignore')` in the first
cell, split download / preprocess / train into separate cells so it is visible where
things stop, and interrupt anything that has shown no change for five minutes.
