# Bird-Eye-View

ECE Paris senior year project.

Aiming at creating a bird-eye view from the surroundings of a car, based on cameras images with segmantic segmentation placed on the car.

## Progress 
- [x] Implementing first model in PyTorch
- [x] Importing and preprocessing data (by using already existing data, or exporting data from Carla on our own)
- [ ] Testing first model
- [ ] Implementing other NN architecture
- [ ] Compare the results
- [ ] Implement other features 

## Issue  
<details>
<summary> - problem to convert the parameter from Keras to Pytorch (especially padding:'same'):</summary> 
__Issue details:__ 'Same' padding means the size of output feature-maps are the same as the input feature-maps (under the assumption of  stride=1 ). For instance, if input is  nin  channels with feature-maps of size  28×28 , then in the output you expect to get  nout  feature maps each of size  28×28  as well. Somehow, Pytorch does not has this option.</br>
__Solution :__</br>
with W:input volume size, F:kernel size, S:stride, P:amount of padding we have this formula output volume = (W-F+2P)/S+1
</details>

## Data

Our Data will be provided by the CARLA simulator.
More details incoming later.

## Model

We are currently rebuilding 3 models from [this repository](https://github.com/MankaranSingh/Auto-Birds-Eye), using PyTorch Library.

<details>
	<summary> Deeper Autoencoder (Davide) (click to expand) </summary>

![AE1](/Images/model_AE1.png)
</details>

<details>
	<summary> Autoencoder (Loïc) (click to expand) </summary>

![AE2](/Images/model_AE2.png)
</details>

<details>
	<summary> Unet (Rémi) (click to expand) </summary>

![Unet](./Images/model_Unet.png)
</details>




By [Davide](https://github.com/Davide-gtr), [Loic](https://github.com/Loicmag) and [Rémi](https://github.com/Wardmisp)
