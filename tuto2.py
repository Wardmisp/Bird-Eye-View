import glob
import os
import sys
from time import sleep
from BEVNetwork import *
from keras.models import load_model

import matplotlib.pyplot as plt

try:
    sys.path.append(glob.glob('./CarlaFile/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import numpy as np
import cv2
import time

actor_list = []

global image_z

from utilBEV import *


def neural_network(model, image):
    print(np.shape(image))
    a = torch.tensor(transfo_image(image), dtype=torch.float).permute(0,3,1,2)
    pred = preprocess(model(a))
  
    return pred


def process_img(image):
    global image_z
    i = np.array(image.raw_data)
    
    i2 = i.reshape((200, 300, 4))
    i3 = i2[...,:3]
    image_z=i3[:,:,2]
    np.save('./test', image_z)
    front_camera = image_z

    return front_camera

try:

       client = carla.Client('localhost', 2000)
       client.set_timeout(10)
       # Once we have a client we can retrieve the world that is currently
       world = client.get_world()
       world = client.load_world("Town01")
       world.set_weather(carla.WeatherParameters.ClearNoon)
       maps = world.get_map()
       
       # The world contains the list blueprints that we can use for adding new
       # actors into the simulation.

       blueprint_library = world.get_blueprint_library()
       model_3 = blueprint_library.filter('model3')[0]
       transform = world.get_map().get_spawn_points()[1]
       vehicle = world.try_spawn_actor(model_3, transform)
       actor_list.append(vehicle)
       vehicle.set_autopilot(True)

       sem_cam = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
       sem_cam.set_attribute('image_size_x', f'{300}')
       sem_cam.set_attribute('image_size_y', f'{200}')
       sem_cam.set_attribute('fov', '110')
       transform = carla.Transform(carla.Location(x=1.2, z=1.5))
       sensor = world.spawn_actor(sem_cam, transform, attach_to=vehicle)
       sensor.listen(lambda data : process_img(data))
       

       actor_list.append(sensor)
       
       # Chargement du réseau de neurones
       name_file = './bev.pth'
       model = get_model()
       model.load_state_dict(torch.load(name_file, map_location=torch.device("cpu")))
       model.eval()
       while True:
            image = image_z
            bev_final=neural_network(model,image)
            #Paramètres permettant de modifier la teinte de la caméra BEV            
            b = 150
            g = 1
            r = 1

            label_hue = np.uint8(b*bev_final/np.max(bev_final))
            blank_ch = g*np.ones_like(label_hue)
            blank_ch_2 = r*np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue,blank_ch, blank_ch_2])

            img_resize=cv2.resize(labeled_img,(500,500))
            cv2.imshow("",img_resize)
            cv2.waitKey(1)

  
            pass

finally:
    for actor in actor_list:
        actor.destroy()


