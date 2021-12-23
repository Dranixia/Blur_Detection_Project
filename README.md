# Blur Detection with Haar Wavelet Transform


## Requirements
* [Python3](https://www.python.org/)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)

Install these using the following command:

```bash
$ pip install -r requirements.txt
```

## Usage

To run the python script with the sample images uploaded to this repo.
```bash
python blur_wavelet.py -i images/blur
```

#### Configuration of edge threshold

The [paper](http://tonghanghang.org/pdfs/icme04_blur.pdf) defines two parameters in order to configure the algorithm. The first is **threshold**. It is used to select if a pixel of Haar transform image is considered as Edge Point. Default value is 35. If you select a smaller threshold, it is more likely an image to be classified as blur.

The default **threshold** is 35. You can define it by adding the parameter in the call:
```bash
python blur_wavelet.py -i images/noblur --threshold 25
```

#### Configuration of decision threshold

In the [paper](http://tonghanghang.org/pdfs/icme04_blur.pdf) it is called **MinZero**. If **Per** is smaller than **MinZero** the image is classified as blur. The default value is 0.001 .
In order to configure the **MinZero** threshold, run the script with the flag **-d**

```bash
python blur_wavelet.py -i images/noblur -d 0.005
```

#### Save results as .JSON

In order to save the output as .JSON, run the script with the flag **-s SAVE_PATH.json** . 

```bash
python blur_wavelet.py -i images/blur -s output.json
```

## Sources

#### Dataset
The sample images have been taken from this [image dataset](https://mklab.iti.gr/results/certh-image-blur-dataset/).


#### Paper

This algorithm is based entirely on this [paper](http://tonghanghang.org/pdfs/icme04_blur.pdf)
