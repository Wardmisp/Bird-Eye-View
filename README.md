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
- problem to convert the parameter from Keras to Pytorch (especially padding:'same') 

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
