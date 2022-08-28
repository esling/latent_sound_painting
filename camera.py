"""

 ~ Latent sound painting // Pompidou center ~
 camera.py : Main class for camera handling
 
 This file defines the overall behavior of the latent control through painitng
 behavior. 
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
 
 All authors contributed equally to the project.
 
"""

import cv2
import numpy as np
from multiprocessing import Event
from pyo import *

class Camera():    
    '''
        The Button class implements interaction with GPIO push buttons. 
        It configures GPIO pins with a callback and asynchronous signal
        to wake up some external processes
    '''
    
    def __init__(self, 
            callback: callable,
            port: int = 0,
            threshold: float = 5e-4,
            fg_detect: str = "KNN",
            optical_flow: str = None
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
        self._port = port
        self._threshold = threshold
        # Setup callback 
        self._callback = callback
        # Create our own event signal
        self._signal = Event()
        # Initialize various images
        self._prev_img = None
        self._cur_img = None
        self._init_img = None
        self._cur_centroid = []
        # Various camera detection modes
        self._movement = False
        self._movement_started = False
        self._movement_stopped = False
        self._fg_detect = fg_detect
        self._optical_flow = optical_flow
        self._pitch = 0
        self._amp = 0
        
    def preprocess_img(self, img):
        """
        Pre-process an image taken from the camera
        """
        return img.astype('float32') / 255.0
    
    def burn_in(self, state):
        """
        Burn-in period to detect base (background image)
        """
        # Read current camera value
        result, img = self._camera.read()
        for i in range(4):
            _, cur_img = self._camera.read()
            img += cur_img
        # Compute base image
        self._init_img = self.preprocess_img(img.astype('float32') / 5.0)
        # Fill the prev image
        self._prev_img = self._init_img
        self._prev_img_raw = cur_img
    
    def detect_movement(self, state):
        """
        Simple movement detector based on delta between successive frames
        """
        if (self._prev_img is not None):
            delta_img = np.abs((self._cur_img - self._prev_img) ** 2)
            if (np.mean(delta_img) > self._threshold):
                if (self._movement == False):
                    self._movement_started = True
                self._movement = True
            else:
                self._movement = False
                self._movement_stopped = True
                self._movement_started = False
            self._delta_img = delta_img
            
    def init_optical_flow(self, state):        
        # params for ShiTomasi corner detection
        self._feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for l/kanade optical flow
        self._lk_params = dict( winSize  = (15, 15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self._color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        self._old_gray = cv2.cvtColor(self._prev_img_raw, cv2.COLOR_BGR2GRAY)
        self._p0 = cv2.goodFeaturesToTrack(self._old_gray, mask = None, **self._feature_params)
        # Create a mask image for drawing purposes
        self._mask = np.zeros_like(self._prev_img)

        
    def optical_flow(self, img):
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self._old_gray, frame_gray, self._p0, None, **self._lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self._p0[st==1]
        else:
            return
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self._mask = cv2.line(self._mask, (int(a), int(b)), (int(c), int(d)), self._color[i].tolist(), 2)
            self._mask = cv2.circle(self._mask, (int(a), int(b)), 5, self._color[i].tolist(), -1)
        # Now update the previous frame and previous points
        self._old_gray = frame_gray.copy()
        self._p0 = good_new.reshape(-1, 1, 2)
        
    
    def plot_images(self, state):
        cv2.imshow("camera", self._cur_img)
        cv2.imshow("delta", self._delta_img)
        if (self._fg_detect is not None):
            cv2.imshow("fg_mask", self._fg_mask)
            cv2.imshow("contours", self._contours)
        if (self._optical_flow is not None):
            cv2.imshow("flow", self._mask)
            
    def detect_contours(self, state):
        gray = cv2.cvtColor(self._cur_img * np.repeat((self._fg_mask / 255.0)[:, :, np.newaxis], 3, axis=2), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 5000)
        flag, thresh = cv2.threshold(np.uint8(blur * 255), 150, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True) 
        # Select long perimeters only
        perimeters = [cv2.arcLength(contours[i], True) for i in range(len(contours))]
        listindex = [i for i in range(min(1, len(perimeters))) if perimeters[i] > perimeters[0]/2]
        # Add contours to image
        [cv2.drawContours(self._contours, [contours[i]], 0, (0,255,0), 5) for i in listindex]
        if len(listindex) > 0:
            self._cur_centroid = np.mean(contours[listindex[0]], axis = 0)
            self._contours = cv2.circle(self._contours, (int(self._cur_centroid[0, 0]), int(self._cur_centroid[0, 1])), 10, [0, 0, 1], -1)
            print(self._cur_centroid)
            self._pitch = self._cur_centroid * 10
            self._amp = self._cur_centroid[0, 1] / 500.0
        else:
            self._cur_centroid = []
            self._pitch = 0
            self._amp = 0

    def read_loop(self, state):
        """
        

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        while True:
            # Read current camera value
            result, img = self._camera.read()
            if result:
                # Pre-process current frame
                self._cur_img = self.preprocess_img(img)
                # Movement detection
                self.detect_movement(state)
                # Foreground detection
                self._fg_mask = np.float32(self._back_sub.apply(img))
                self._contours = self._cur_img.copy()
                #if (self._movement):
                    # Perform contour detection
                self.detect_contours(state)
                # Optical flow
                if (self._optical_flow is not None):
                    if (self._movement_started):
                        self.init_optical_flow(state)
                        self._movement_started = False
                    self.optical_flow(img)
                # Record previous image
                self._prev_img = self._cur_img
                self._prev_img_raw = img
                # Plot all images
                self.plot_images(state)
                # Handle keyboard inputs
                k = cv2.waitKey(1)
                if k == 'q':
                    break
            else:
                print("Failed")
                break
            
    def callback(self, state, queue, delay=0.001):
        """
            Function for reading the current CV values.
            Also updates the shared memory (state) with all CV values
            Parameters:
                state:      [Manager]
                            Shared memory through a Multiprocessing manager
                queue:      [Queue]
                            Shared memory queue through a Multiprocessing queue
                delay:      [int], optional
                            Specifies the wait delay between read operations [default: 0.001s]
        """
        # Main reading loop
        try:
            # Create the camera object
            self._camera = cv2.VideoCapture(self._port)
            # Create windows
            cv2.namedWindow("camera", cv2.WINDOW_AUTOSIZE)
            # Compute base image
            self.burn_in(state)
            # Init detectors
            if (self._fg_detect is not None):
                cv2.namedWindow("fg_mask", cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow("contours", cv2.WINDOW_AUTOSIZE)
                if (self._fg_detect == 'MOG2'):
                    self._back_sub = cv2.createBackgroundSubtractorMOG2()
                else:
                    self._back_sub = cv2.createBackgroundSubtractorKNN()
            # Init optical flow
            if (self._optical_flow is not None):
                cv2.namedWindow("flow", cv2.WINDOW_AUTOSIZE)
                self.init_optical_flow(state)
            # Read loop
            self.read_loop(state)
        except KeyboardInterrupt:
            pass

    def __del__(self):
        '''
            Destructor - cleans up the GPIO.
        '''
        cv2.destroyAllWindows()
        self._camera.release()
