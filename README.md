# manifold-net-dmri
[ManifoldNet Paper](https://github.com/cvgmi/manifold-net-dmri/raw/master/manifoldNet.pdf) (published in the International conf. on IPMI 2019) Implementation for Diffusion MRI (dMRI)

Chakraborty R., Bouza J., Manton J., Vemuri B.C. (2019) A Deep Neural Network for Manifold-Valued Data with Applications to Neuroimaging. In: Chung A., Gee J., Yushkevich P., Bao S. (eds), Proceedings of the International Conference on Information Processing in Medical Imaging. (IPMI) 2019. Lecture Notes in Computer Science, vol 11492. Springer, Cham

## Dependencies
- Pytorch 1.1
- [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)

By default, ```batch_svd.py``` is expected to be put in the same directory as the source code.

## Quickstart
We are not currently including our in-house dataset. For this reason you will need to include your own DTI (Diffusion Tensor Image) dataset and modify the dataloader defined in ```model.py``` accordingly. 

Once the data is ready, simply run ```train.py```.
