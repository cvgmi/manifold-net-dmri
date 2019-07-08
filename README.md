# manifold-net-dmri
[ManifoldNet Paper](https://www.cise.ufl.edu/~vemuri/paperphp/article.php?y=2019&i=2) Implementation for Diffusion MRI (dMRI)

## Dependencies
- Pytorch 1.1
- [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)

## Quickstart
We are not currently including our in-house dataset. For this reason you will need to include your own DTI (Diffusion Tensor Image) dataset and modify the dataloader defined in ```model.py``` accordingly. 

Once the data is ready, simply run ```train.py```.
