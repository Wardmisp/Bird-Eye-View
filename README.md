# Bird-Eye-View

ECE Paris senior year project.

Aiming at creating a bird-eye view from the surroundings of a car, based on cameras images with segmantic segmentation placed on the car.

## Progress 
- [x] Implementing first model in PyTorch
- [x] Importing and preprocessing data (by using already existing data, or exporting data from Carla on our own)
- [x] Testing first model
- [x] Implementing other NN architecture
- [x] Compare the results
- [ ] Implement other features 

## Issue  
<details>
<summary> - problem to convert the parameter from Keras to Pytorch (especially padding:'same'):</summary> 
__Issue details:__ 'Same' padding means the size of output feature-maps are the same as the input feature-maps (under the assumption of  stride=1 ). For instance, if input is  nin  channels with feature-maps of size  28×28 , then in the output you expect to get  nout  feature maps each of size  28×28  as well. Somehow, Pytorch does not has this option.</br>
__Solution :__</br>
with W:input volume size, F:kernel size, S:stride, P:amount of padding we have this formula output volume = (W-F+2P)/S+1
</details>

<details>
<summary> - The output is noisy:</summary> 
__Issue details:__ The output of the current network is too noisy to be properly used.</br>
__Solution :__</br>
Try to change the network architecture.
</details>

## Data

Our Data will be provided by the CARLA simulator.
More details incoming later.

## Model

We are currently rebuilding 3 models from [this repository](https://github.com/MankaranSingh/Auto-Birds-Eye), using PyTorch Library.

<details>
	<summary> Deeper Autoencoder (Loïc) (click to expand) </summary>

![AE1](/Images/model_AE1.png)
</details>

<details>
	<summary> Autoencoder (Davide) (click to expand) </summary>

![AE2](/Images/model_AE2.png)
</details>

<details>
	<summary> Unet (Rémi) (click to expand) </summary>
![Unet](./Images/model_Unet.png)</br>

Comparaison après implémentation en PyTorch/Implémentation initiale avec Keras:</br>
![UnetInfo](./Images/Unet_info.JPG)</br>
![UnetInfo0](./Images/First_unet_summary.JPG)
</details>

## Loss Functions
<details>
	
<summary> 
	### SSIM (Structural Similarity Index) 
</summary>
It is a metric that measures the structural similarity of two images (rather than a pixel-to-pixel difference). It is used as a “loss function”, taking into account luminance, contrast and structure. Used to measure the quality of a compressed image compared to the original image. Aims to reproduce human vision.
Performance: Appears to be imprecise (less than expected) and intended as a measure of still image quality.
</details>

<details>
<summary>
	Dice-Coefficient Loss (region-based)
</summary>
This coefficient is a statistical indicator that measures the similarity between two samples. Often compared to Cross-Entropy: the goal is to maximize the measurement of the Dice coefficient. Cross entropy is only an approximation and is easier to maximize using backpropagation. In addition, the Dice coefficient performs better for class imbalance problems by design (this is a classification problem: the classes are not represented equally, which increases the learning difficulties of the algorithm).
</details>

<details>
<summary>	
	Cross-Entropy Loss (distribution-based)
</summary>
Measures the performance of a model whose output is a probability value between 0 and 1, by measuring the distance between the predicted value and the true value. The more the predicted value deviates from the real value, the more the “Cross-Entropy Loss” increases: thus, a perfect model would have a loss of 0. The score associated with each probability is calculated from a logarithm: thus, the higher the large differences close to 1 the score is high and the small differences close to 0 obtain low scores.

In general, we cannot predict which function will be the most efficient on a particular set of data, so the best solution is to test them all and compare the results. 
</details>

<details>
<summary>
	Custom Loss-Function
</summary>
Defined under the name "Custom_loss", this is a Loss Function created by the author of the Github and declared as a combination of the Dice-Coefficient and the SSIM:
Custom_loss = Dice_coef + 5 * SSIM_loss
An explanation can come from the architecture of Autoencoders. Ideally, an Autoencoder model strikes a balance between:
- A sensitivity to the input data to reconstruct the representation in a fairly precise way.
- Insensitivity to the input data to "discourage" the model from memorizing the inputs and therefore to avoid overloading.
Thus, the model is forced to keep only the data variations necessary for the reconstruction of the image and to avoid redundancies. To do this, we must build a Loss Function with a term that sensitizes the model to the input data (here the Dice_coef) and we add a term to discourage the model called "regularizer" (here the SSIM_loss). In addition, we introduce a scale factor in front of the regularizer to manage the balance between the two objectives (here we use 5) [1].

[1] Jeremy Jordan «Introduction to Autoencoders». 19 March 2018. Jeremyjordan.me
https://www.jeremyjordan.me/autoencoders/
</details>

## Intégration à Carla (Windows)

Une fois que les réseaux de neurones ont été retranscrits et entraînés, nous avons souhaité les intégrer au simulateur Carla pour obtenir une évolution en temps réel de la Bird-Eye View. Afin de reproduire cette caméra, il suffit de télécharger les fichiers suivants et de les placer dans un même dossier : 
- "BEV_carla.py" script Python permettant d'afficher la Bird-Eye View.
- "utilBEV.py" script Python permettant d'appliquer les transformations nécessaires sur les données en entrée du réseau.
- "autoencoder.py" script Python contenant le réseau de neurone de type Autoencoder.

Ensuite il vous faut lancer le simulateur Carla et attendre que le logiciel se charge complètement.
Une fois que Carla est lancé, ouvrez l'invite de commande (cmd) et placez vous dans le dossier où se trouvent vos scripts téléchargés.
Finalement, lancez la commande suivante : "python BEV_carla.py".

Attention : il arrive très souvent que cela ne marche pas du premier coup (erreur de connexion à Carla), il vous faut alors réessayer plusieurs fois avant que ça marche.


By [Davide](https://github.com/Davide-gtr), [Loic](https://github.com/Loicmag) and [Rémi](https://github.com/Wardmisp)
