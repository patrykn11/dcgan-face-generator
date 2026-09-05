# DCGAN Face Generator

A TensorFlow/Keras DCGAN trained on the CelebA dataset to generate new 64x64 RGB face images from random 100-dimensional latent vectors.

- **Data pipeline:** loads CelebA, crops and resizes the images, normalizes them to [-1, 1], and creates batches.
- **Model implementation:** builds a convolutional generator and discriminator.
- **Training:** uses binary cross-entropy, Adam optimizers, label smoothing, and custom gradientTape updates.
- **Evaluation:** calculates Inception Score, FID, and discriminator accuracy.

## Generated faces after epoch 20

![Generated faces after epoch 20](face_gan_output/faces_epoch_020.png)
