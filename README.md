# manifold-net-dmri
[ManifoldNet Paper](https://www.cise.ufl.edu/~vemuri/paperphp/article.php?y=2019&i=2) (published in IPMI'19) Implementation for Diffusion MRI (dMRI)

Chakraborty R., Bouza J., Manton J., Vemuri B.C. (2019) A Deep Neural Network for Manifold-Valued Data with Applications to Neuroimaging. In: Chung A., Gee J., Yushkevich P., Bao S. (eds) Information Processing in Medical Imaging. IPMI 2019. Lecture Notes in Computer Science, vol 11492. Springer, Cham

## Dependencies
- Pytorch 1.1
- [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)

By default, ```batch_svd.py``` is expected to be put in the same directory as the source code.

## Quickstart
We are not currently including our in-house dataset. For this reason you will need to include your own DTI (Diffusion Tensor Image) dataset and modify the dataloader defined in ```model.py``` accordingly. 

Once the data is ready, simply run ```train.py```.
