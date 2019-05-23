## HiLLoC: Lossless Image Compression with Hierarchical Latent Variable Models
Code for reproducing the results to the NeurIPS submission "HiLLoC: Lossless Image Compression with Hierarchical Latent Variable Models"

We adapt open source implementations for the Resnet VAE (https://github.com/openai/iaf) and PixelVAE (https://github.com/igul222/PixelVAE).

The compression is implemented using [Craystack](craystack).

### Training the models
 * RVAE. To train 24-layer model: `python --hpconfig depth=1,num_blocks=24,kl_min=0.1,learning_rate=0.002,batch_size=32,enable_iaf=False,dataset=<cifar10|small32_imagenet|small64_imagenet> --num_gpus 1 --mode train --logdir <log_path>`
 * PixelVAE. To train the two layer model for ImageNet64: `python pvae/train.py <data_path> --dataset imagenet_64 --settings 64px_big`

### Running compression
 * RVAE. To use a 24-layer model: `python --hpconfig depth=1,num_blocks=24,kl_min=0.1,learning_rate=0.002,batch_size=32,enable_iaf=False,dataset=<test_images|cifar10|small32_imagenet|small64_imagenet>,compression_exclude_sizes=True --num_gpus 1 --mode bbans --evalmodel <directory name of model to evaluate relative to <log_path>/train/>`
 * PixelVAE. To use the two layer model `python pvae/pixelvae_bbans_two_layer.py <data_path> --load_path <path_to_model> --dataset imagenet_64 --settings 64px_big`
