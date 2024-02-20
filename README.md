# WGAN: Wasserstein GAN
The Wasserstein GAN was implemented based on a linear activation in the discriminator (now called critic!) and a gradient penality.
This penality term enforces the critic to be Lipschitz-continuous.

## Evaluation

### Loss

<img src="./plots/Loss_GradientPenality.png" width="650" height="200">

### Generated images while training
The WGAN was trained for 300 epochs:
Each column represents a noise vector which is transformed by the generator into an image.
There are ten different noise vectors.
The i^th row represents the model's state at epoch i * 10.

<img src="./plots/GeneratedImgsWhileTraining.png" width="650" height="2000">

### Generated images
Let's generate 100 images after training the GAN i.e. after 300 epochs.

<img src="./plots/GenerateImgs.png" width="650" height="650">
