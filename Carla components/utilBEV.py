import numpy as np


#############################################################*
# Transformation de l'image d'entree

def transfo_image(img, nb_label_class=[0, 1, 6, 7, 8, 10, 11]):
	img = img[::2, ::2]
	img = img[2:-2, 3:-3]
	img = img_to_categorical(img, nb_label_class)

	return np.expand_dims(img, axis=0)



#############################################################
# Setting up data generators

def img_to_categorical(img, needed_labels):
  
  cat = np.empty((img.shape[0], img.shape[1], len(needed_labels)))

  for channel, label in enumerate(needed_labels):
    cat[:, :, channel] = np.where(np.isin(img, label), 1, 0)

  return cat

def categorical_to_img(cat):
  
  img = np.argmax(cat, axis=-1)
  return img


def preprocess(img):
  output_numpy = img.squeeze(0).permute(1,2,0).detach().numpy()
  output_final = np.rot90(categorical_to_img(output_numpy),3)
  return output_final