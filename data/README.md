# ImageArg — dataset notes

**Source:** *ImageArg: A Multi-modal Tweet Dataset for Image Persuasiveness Mining* (Liu et al., ARGMINING 2022). Code & annotated files: MeiqiGuo/ArgMining2022-ImageArg on [`GitHub`](https://github.com/ImageArg/ImageArg-Shared-Task).

## What ImageArg is (high-level)
ImageArg is a multimodal dataset of tweets (text + image) created to study how images affect the persuasiveness of argumentative social media content. The dataset was developed with a careful annotation pipeline that first labels stance, and then (for tweets with a clear stance) annotates image persuasiveness, image content type, and image persuasion mode (logos/pathos/ethos).

## Key facts

- **Topics covered (collection):** gun control (main annotated split), with pilot data for immigration and abortion. 
- **Raw collection size:** ~286,000 tweets (collected via Twitter API between 2019-03-29 and 2021-03-29) before filtering.
- **Annotated (final) dataset (first release):** 1,003 annotated samples for the gun control topic (these are tweets with a clear support/oppose stance and additional image annotations). Pilot samples exist for immigration and abortion but the primary, released corpus is the gun control subset.
- **Annotation dimensions:**
  - **Stance:** support / oppose / neutral / irrelevant (stance annotation required first). 
  - **Image Persuasiveness (IP):** continuous persuasiveness score computed as the difference between text-only and text+image persuasiveness ratings by annotators; can be thresholded to produce binary labels (persuasive / not persuasive).
  - **Image content types:** statistics, testimony, anecdote, slogan, scene photo, symbolic photo, other. 
  - **Persuasion mode:** logos (logic), pathos (emotion), ethos (credibility).
- **Distribution notes:** In the gun-control subset the stance labels are roughly balanced (≈46% support, 54% oppose). About 259 samples were annotated as persuasive images (using the authors' chosen threshold).

## How to use (important practical notes)
- The GitHub repository contains the annotated JSON (`data/gun_control.json`) and the code to run the baselines.
- The repo code includes example scripts for:
  - text-only experiments (`main_text.py`)
  - image-only experiments (`main_image.py`)
  - multimodal experiments (`main_multimodality.py`).

## References
- Liu et al. (2022). *ImageArg: A Multi-modal Tweet Dataset for Image Persuasiveness Mining*.  
- Liu et al. (2023). *Overview of ImageArg-2023: The First Shared Task in Multimodal Argument Mining*.

See the full citation in [`REFERENCES.bib`](../REFERENCES.bib).
