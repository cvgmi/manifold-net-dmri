# manifold-net-dmri
[ManifoldNet Paper](https://github.com/cvgmi/manifold-net-dmri/raw/master/manifoldNet.pdf) (published in the International conf. on IPMI 2019) Implementation for Diffusion MRI (dMRI)

Please cite the following papers if you use this code: 

Chakraborty, R., Bouza, J., Manton, J., & Vemuri, B. C. (2020). Manifoldnet: A deep neural network for manifold-valued data with applications. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Chakraborty R., Bouza J., Manton J., Vemuri B.C. (2019) A Deep Neural Network for Manifold-Valued Data with Applications to Neuroimaging. In: Chung A., Gee J., Yushkevich P., Bao S. (eds), Proceedings of the International Conference on Information Processing in Medical Imaging. (IPMI) 2019. 

## Dependencies
- Pytorch 1.1

## Quickstart
We are not currently including our in-house dataset. For this reason you will need to include your own DTI (Diffusion Tensor Image) dataset and modify the dataloader defined in ```model.py``` accordingly. 

Once the data is ready, simply run ```train.py```. Included under `data/example.npy` is an example dataset with 4 synthetically generated SPD-valued images. By default, running `train.py` will overfit a model to these samples, you will see training loss approaching 0. 
