# Hungarian Patch Swap

A cute algorithim for style transfer

## Files

- `style_patch_match.py`: Performs style transfer using patch-based matching.
- `README.md`: This file.
- `requirements.txt`: Lists Python dependencies.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

This code is very exipsnive to run for combinations of large img_height/img_width with small patch sizes, ive tested with good quick results for 512 Width/Height and 8x8 patches. 

More instructions coming soon :)

```bash
python style_patch_match.py --style path/to/style.jpg --content path/to/content.jpg --img_height 256 --img_width 256 --patch_h 8 --patch_w 8 --feature_type greyscale --comparison_metric l2 --out_dir output
```

## Demo Images

These are examples of some images stylized with a picture of lorem ipsum text


<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">

<figure style="text-align: center; width: 180px;">
  <img src="lorem_ipsum_style_demo/lorem_ipsum.jpg" alt="Lorem Ipsum" style="width: 100%; height: auto;">
  <figcaption>Lorem Ipsum</figcaption>
</figure>

<figure style="text-align: center; width: 180px;">
  <img src="lorem_ipsum_style_demo/stylized_charizard_from_lorem_ipsum_greyscale_l2_8x8.png" alt="Stylized Charizard" style="width: 100%; height: auto;">
  <figcaption>Stylized Charizard</figcaption>
</figure>

<figure style="text-align: center; width: 180px;">
  <img src="lorem_ipsum_style_demo/stylized_rizzler_from_lorem_ipsum_greyscale_l2_8x8.png" alt="Stylized Rizzler" style="width: 100%; height: auto;">
  <figcaption>Stylized Rizzler</figcaption>
</figure>

<figure style="text-align: center; width: 180px;">
  <img src="lorem_ipsum_style_demo/stylized_snow_from_lorem_ipsum_greyscale_l2_8x8.png" alt="Stylized Snow" style="width: 100%; height: auto;">
  <figcaption>Stylized Snow (my cat)</figcaption>
</figure>

<figure style="text-align: center; width: 180px;">
  <img src="lorem_ipsum_style_demo/stylized_starry_night_from_style_greyscale_l2_8x8.png" alt="Stylized Starry Night" style="width: 100%; height: auto;">
  <figcaption>Stylized Starry Night</figcaption>
</figure>

</div>
