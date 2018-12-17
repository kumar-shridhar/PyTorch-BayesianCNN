# Superresolution using an efficient sub-pixel convolutional neural network

This example illustrates how to use the efficient sub-pixel convolution layer described in  ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution within your network for tasks such as superresolution.

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED] [--resume RESUME]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
  --resume              resume from checkpoint
```

#### Training on [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

	Put root_dir = download_bsd300() in data.py file

#### Training on your own dataset:

	* Put root_dir = document_dataset() in data.py file
	* Use the following folder structure:
		   dataset
		   |
		    --- document
		        |
		         --- images
		             |
		              --- test
		             |
		              --- train 


#### A snapshot of the model after every epoch with be saved as filename model_epoch_<epoch_number>.pth

## Example Usage:
 
### Train

`python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`

### Super Resolve

`python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png`

### Resume from Checkpoints

`provide path with --resume arguement`

## Create dataset by downloading HD images from Google

Use the script `scrape_google_search_images.py` to scrape HD images from Google search results

```
usage: scrape_google_search_images.py [--search SEARCH] [--num_images NUM_IMAGES] [--directory DIRECTORY]

Scrape Google images

arguments:
  --search              search term
  --num_images          number of images to save
  --directory           directory path to save results
```

Point to Note: Script `scrape_google_search_images.py` works well with Python 2.x version.