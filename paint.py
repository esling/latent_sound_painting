"""

 ~ Latent sound painting // Pompidou center ~
 camera.py : Main class for camera handling
 
 This file defines the overall behavior of the latent control through painitng
 behavior. 
 
 Author               :  Philippe Esling
                        <esling@google.com>
 
 All authors contributed equally to the project.
 
"""

import numpy as np
from multiprocessing import Event
from dataclasses import dataclass
from typing import Array

@dataclass
class Line():
    color: np.ndarray
    points: np.ndarray
    

class Paint():    
    '''
        The Button class implements interaction with GPIO push buttons. 
        It configures GPIO pins with a callback and asynchronous signal
        to wake up some external processes
    '''
    lines: Array[Line]
    
    def __init__(self, 
            callback: callable,
            ):
        '''
            Constructor - Creates a new instance of the Navigation class.
            Parameters:
                callback:   [callable]
                            Outside function to call on button push
                port:       [int], optional
                            Specify the camera port [default: 0]
                debounce:   [int], optional
                            Debounce time to prevent multiple firings [default: 250ms]
        '''
        # Setup callback 
        self._callback = callback
        # Create our own event signal
        self._signal = Event()
        # Create structures
        self._signal = Event()
        
    def create_image(self):
        return
        
