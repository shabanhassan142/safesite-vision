# Dataset Report — safesite-vision

## Split Summary

| Split | Images | With boxes | Background |
|-------|--------|------------|------------|
| train | 2605 | 2535 | 70 |
| valid | 114 | 84 | 30 |
| test | 82 | 60 | 22 |

## Box Counts per Class per Split

| ID | Class | train | valid | test | Total |
|----|-------|-------|-------|------|-------|
| 0 | person | 9532 | 166 | 174 | 9872 |
| 1 | hardhat | 3145 | 79 | 110 | 3334 |
| 2 | no_hardhat | 2317 | 69 | 41 | 2427 |
| 3 | safety_vest | 3033 | 41 | 61 | 3135 |
| 4 | no_safety_vest | 3962 | 106 | 90 | 4158 |

## Class Imbalance

- **person**: 9872 boxes (43.1%)
- **hardhat**: 3334 boxes (14.5%)
- **no_hardhat**: 2427 boxes (10.6%)
- **safety_vest**: 3135 boxes (13.7%)
- **no_safety_vest**: 4158 boxes (18.1%)

## Near-Duplicate Removal

- Valid images removed (dup of train): **0**
- Test images removed (dup of train): **0**
- Near-duplicate groups inside train: **0**

## Augmentation Analysis

| Split | Total | Mosaic | Grayscale | Cutout |
|-------|-------|--------|-----------|--------|
| train | 2605 | 2410 | 242 | 1601 |
| valid | 114 | 108 | 0 | 61 |
| test | 82 | 77 | 0 | 45 |

> Mosaic: ~92% of images appear to be 4-photo stitches (YOLOv5 augmentation).
> Grayscale: only present in train (242 images, ~9%).
> Cutout: black-patch augmentation present in ~61% of train, ~54% of valid/test.
> Valid and test are also augmented — consider whether a clean held-out test set is needed.

## Sample Images (train, with boxes)

![IMG_5024_mp4-2_jpg.rf.ef79470223ea72c2c2a7604795c2d3da](reports/sample_images/IMG_5024_mp4-2_jpg.rf.ef79470223ea72c2c2a7604795c2d3da.jpg)
![Movie-on-10-31-22-at-10_08-AM_mov-0_jpg.rf.5eb18cc65358b70ee602eafa99f2ff54](reports/sample_images/Movie-on-10-31-22-at-10_08-AM_mov-0_jpg.rf.5eb18cc65358b70ee602eafa99f2ff54.jpg)
![1288788-une-employee-aide-des-voyageurs-en-provenance-de-chine-le-26-janvier-2020-a-l-aeroport-de-roissy_jpg.rf.93dd6595c9b6f435a03af2e09588744c](reports/sample_images/1288788-une-employee-aide-des-voyageurs-en-provenance-de-chine-le-26-janvier-2020-a-l-aeroport-de-roissy_jpg.rf.93dd6595c9b6f435a03af2e09588744c.jpg)
![image_153_jpg.rf.da2dfd3346476f74fa1ddc0690c5defc](reports/sample_images/image_153_jpg.rf.da2dfd3346476f74fa1ddc0690c5defc.jpg)
![youtube-344_jpg.rf.86d332d55a3bc8381c851f1386fdd010](reports/sample_images/youtube-344_jpg.rf.86d332d55a3bc8381c851f1386fdd010.jpg)