"""

 ~ Latent sound painting // Pompidou center ~
 test.py : Main class for testing purposes
 
 This file defines the overall behavior of the latent control through painitng
 behavior. 
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
 
 All authors contributed equally to the project.
 
"""

import argparse
import multiprocessing as mp
from multiprocessing import Process, Manager, Queue
# Library imports
from camera import Camera

def callback_camera():
    print("Camera event")

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument(
        "--cam_port",
        default=0,      
        type = int,   
        help="Camera port for acquisition")
    parser.add_argument(
        "--thresh_detect",       
        default=1e-3,
        type = str,       
        help = "Threshold for movement detection")
    parser.add_argument(
        "--algo", 
        type = str, 
        help = "Background subtraction method (KNN, MOG2)", 
        default='MOG2')
    # Parse the arguments
    args = parser.parse_args()
    
    # Create the camera
    camera = Camera(callback_camera)
    # List of objects to create processes
    objects = [camera]
    # Find number of CPUs
    nb_cpus = 4
    # Create a pool of jobs
    pool = mp.Pool(nb_cpus)
    # Handle signal informations
    manager = Manager()
    state = manager.dict()
    state['global'] = manager.dict()
    # Create a queue for sharing information
    queue = Queue()
    processes = []
    for o in objects:
        processes.append(Process(target=o.callback, args=(state, queue)))
    # Start all processes
    for p in processes:
        p.start()
    # Wait all processes
    for p in processes:
        p.join()