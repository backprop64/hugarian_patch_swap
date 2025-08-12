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

<div style="display: flex; flex-wrap: nowrap; gap: 10px; overflow-x: auto;">

  <div style="text-align: center;">
    <img src="lorem_ipsum_style_demo/lorem_ipsum.jpg" alt="Lorem Ipsum" width="180">
    <div>Lorem Ipsum</div>
  </div>

  <div style="text-align: center;">
    <img src="lorem_ipsum_style_demo/stylized_charizard_from_lorem_ipsum_greyscale_l2_8x8.png" alt="Stylized Charizard" width="180">
    <div>Stylized Charizard</div>
  </div>

  <div style="text-align: center;">
    <img src="lorem_ipsum_style_demo/stylized_rizzler_from_lorem_ipsum_greyscale_l2_8x8.png" alt="Stylized Rizzler" width="180">
    <div>Stylized Rizzler</div>
  </div>

  <div style="text-align: center;">
    <img src="lorem_ipsum_style_demo/stylized_snow_from_lorem_ipsum_greyscale_l2_8x8.png" alt="Stylized Snow" width="180">
    <div>Stylized Snow</div>
  </div>

  <div style="text-align: center;">
    <img src="lorem_ipsum_style_demo/stylized_starry_night_from_style_greyscale_l2_8x8.png" alt="Stylized Starry Night" width="180">
    <div>Stylized Starry Night</div>
  </div>

</div>
