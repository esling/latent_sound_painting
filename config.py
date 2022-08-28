# -*- coding: utf-8 -*-
"""

 ~ Latent sound painting // Pompidou center ~
 config.py : Main class for global configuration values
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
 
 All authors contributed equally to the project.
 
"""

class config:
    # Global program info
    class painter:
        device      = 'Latent sound painter'
        version     =  0.01
    
    # Global audio info
    class audio:
        # Screen modes
        mode_idle   = 0
        mode_burnin = 1
        mode_play   = 2
        mode_rec    = 3
        mode_busy   = 4
        # General screen properties
        volume      = 1.0
        stereo      = 0.0
    
    class events:
        none        = -1
        button      = 0
        rotary      = 1
        gate0       = 2
        gate1       = 3
        cv2         = 4
        cv3         = 5
        cv4         = 6
        cv5         = 7
    