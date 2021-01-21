from BEVNetwork import *
from utilBEV import *
import matplotlib.pyplot as plt
import cv2

# Chargement du modèle de réseau de neurone
name_file = './pytorch_model.pth'
model = get_model()
model = torch.load(name_file, map_location=torch.device("cpu"))

model.eval()

# Chargement de l'image à donner en entrée (ici : issu du dataset)
image_dataset = np.load('./data/1.npy')

image_dataset_preprocess = torch.tensor(transfo_image(image_dataset), dtype=torch.float).permute(0,3,1,2)
print(image_dataset_preprocess.shape)
output_dataset = preprocess(model(image_dataset_preprocess))


# Chargement de l'image à donner en entrée (ici : issu du simulateur Carla)
image_carla = np.load('./test.npy')
image_carla_preprocess = torch.tensor(transfo_image(image_carla), dtype=torch.float).permute(0,3,1,2)
output_carla = preprocess(model(image_carla_preprocess))



# On montre les différents résultats
fig=plt.figure()
fig.add_subplot(2,2,1)
plt.imshow(image_dataset) ## image dataset
fig.add_subplot(2,2,2)
plt.imshow(output_dataset, vmin=0, vmax=6) ## bev dataset
fig.add_subplot(2,2,3)
plt.imshow(image_carla) ## image carla
fig.add_subplot(2,2,4)
plt.imshow(output_carla, vmin=0, vmax=6) ## bev carla
plt.show()



# Conversion de l'image pour l'afficher avec CV2
label_hue = np.uint8(179*output_carla/np.max(output_carla))
blank_ch = np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

image_resize=cv2.resize(labeled_img,(250,250))

# Affichage avec CV2
cv2.imshow("",image_resize)
cv2.waitKey(1)
time.sleep(5)
