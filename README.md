# Gelsight Depth Map Reconstruction

This repository contains the code to reconstruct depth maps from RGB images captured using a Gelsight tactile sensor. The implementation is in Python, and it leverages various computer vision and deep learning techniques to accomplish this task. The repository contains data processing, model training, and depth map reconstruction components.

## Directory Structure

```
.
├── data/                   # Data folder containing the dataset
├── model/                  # Model folder containing saved models
├── 1_pixel_unit.py         # Script to define the pixel unit for the sensor
├── 2_dataset.py            # Script to preprocess and prepare the dataset
├── 3_train.py              # Script to train the depth map reconstruction model
├── 4_reconstruction.py      # Script to reconstruct depth maps from trained model
├── 5_reconstruction_test.py # Script to test the reconstruction performance
├── affine_transform.py      # Utility script for affine transformations
├── capture.py               # Utility script for capturing images
├── circle_mask.py           # Utility script for applying circular masks
├── fast_possion.py          # Utility script for fast Poisson equation solver
├── gradient.py              # Utility script for computing image gradients
├── marker_mask.py           # Utility script for creating marker masks
├── model.py                 # Script containing the depth map reconstruction model
└── plot.py                  # Utility script for plotting and visualizing data
```

## Requirements

- Python 3.x
- Pytorch
- NumPy
- OpenCV
- Matplotlib
- Scikit-Image

You can install these dependencies using `pip`:

```bash
pip install pytorch numpy opencv-python matplotlib scikit-image
```

## Usage

1. Place your dataset in the `data/` folder. The dataset should contain RGB images captured using the Gelsight tactile sensor.

2. Define the pixel unit for the sensor using `1_pixel_unit.py`. Run the script as follows:

```bash
python 1_pixel_unit.py
```

3. Preprocess the image using `circle_mask.py` and prepare the dataset using `2_dataset.py`.

4. Train the depth map reconstruction model using `3_train.py`. Run the script as follows:

```bash
python 3_train.py
```

The trained model will be saved in the `model/` folder.

5. Reconstruct depth maps from the trained model using `4_reconstruction.py`. Run the script as follows:

```bash
python 4_reconstruction.py
```

6. Test the reconstruction performance using `5_reconstruction_test.py`. Run the script as follows:

```bash
python 5_reconstruction_test.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

We would like to acknowledge and thank the researchers and developers who contributed to the Gelsight technology and the open-source libraries used in this project.

