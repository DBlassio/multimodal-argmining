# ImageArg — dataset notes

**Source:** *ImageArg: A Multi-modal Tweet Dataset for Image Persuasiveness Mining* (Liu et al., ARGMINING 2022). Code & annotated files: MeiqiGuo/ArgMining2022-ImageArg on [`GitHub`](https://github.com/ImageArg/ImageArg-Shared-Task).

## What ImageArg is (high-level)
ImageArg is a multimodal dataset of tweets (text + image) created to study how images affect the persuasiveness of argumentative social media content. The dataset was developed with a careful annotation pipeline that first labels stance, and then (for tweets with a clear stance) annotates image persuasiveness, image content type, and image persuasion mode (logos/pathos/ethos). :contentReference[oaicite:2]{index=2}

## Key facts (you can paste these into your README)
- **Topics covered (collection):** gun control (main annotated split), with pilot data for immigration and abortion. :contentReference[oaicite:3]{index=3}  
- **Raw collection size:** ~286,000 tweets (collected via Twitter API between 2019-03-29 and 2021-03-29) before filtering. :contentReference[oaicite:4]{index=4}  
- **Annotated (final) dataset (first release):** 1,003 annotated samples for the gun control topic (these are tweets with a clear support/oppose stance and additional image annotations). Pilot samples exist for immigration and abortion but the primary, released corpus is the gun control subset. :contentReference[oaicite:5]{index=5}  
- **Annotation dimensions:**
  - **Stance:** support / oppose / neutral / irrelevant (stance annotation required first). :contentReference[oaicite:6]{index=6}  
  - **Image Persuasiveness (IP):** continuous persuasiveness score computed as the difference between text-only and text+image persuasiveness ratings by annotators; can be thresholded to produce binary labels (persuasive / not persuasive). :contentReference[oaicite:7]{index=7}  
  - **Image content types:** statistics, testimony, anecdote, slogan, scene photo, symbolic photo, other. :contentReference[oaicite:8]{index=8}  
  - **Persuasion mode:** logos (logic), pathos (emotion), ethos (credibility). :contentReference[oaicite:9]{index=9}
- **Distribution notes:** In the gun-control subset the stance labels are roughly balanced (≈46% support, 54% oppose). About 259 samples were annotated as persuasive images (using the authors' chosen threshold). See the original paper/GitHub for full distributions and inter-annotator agreement statistics. :contentReference[oaicite:10]{index=10}

## How to use (important practical notes)
- The GitHub repository contains the annotated JSON (`data/gun_control.json`) and the code to run the baselines. The repo expects that you **download tweet content (text + media) via the Twitter API** into `data/` (the authors provide tweet IDs / references rather than hosting raw Twitter content, in compliance with Twitter terms). See the project repo for scripts and flags. :contentReference[oaicite:11]{index=11}
- **Do not publish** raw tweet content or images unless you follow Twitter's terms of service. For reproducibility, keep the authors' annotations in your `data/` folder and fetch the tweet content yourself as required.
- The repo code includes example scripts for:
  - text-only experiments (`main_text.py`)
  - image-only experiments (`main_image.py`)
  - multimodal experiments (`main_multimodality.py`). :contentReference[oaicite:12]{index=12}

## References

- Z. Liu, M. Guo, Y. Dai and D. Litman (2022). *ImageArg: A Multi-modal Tweet Dataset for Image Persuasiveness Mining*. Proceedings of the 9th Workshop on Argument Mining. See the full citation in [`REFERENCES.bib`](./REFERENCES.bib).
