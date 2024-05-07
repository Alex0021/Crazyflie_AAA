import logging
import sys
import time
from threading import Event
from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper


URI = uri_helper.uri_from_env(default="radio://0/40/2M/E7E7E7E704")


class KeyboardController:
    def __init__(self):
        # Create a Crazyflie object
        self._cf = Crazyflie(rw_cache='./cache')

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self.in_air = False

        print('Connecting to %s' % URI)

        self._cf.open_link(URI)

        self._waiting_for_connection = True
        self.is_connected = False

    def _connected(self, link_uri):
        print('Connected to %s' % link_uri)
        self._waiting_for_connection = False
        self.is_connected = True

        # Create a MotionCommander object
        self._mc = MotionCommander(self._cf)

        # Start the keyboard listener
        listener = keyboard.Listener(on_press=self._on_key_pressed,
                                on_release=self._on_key_released)
        listener.start()

    def _disconnected(self, link_uri):
        print('Disconnected from %s' % link_uri)
        self.is_connected = False
        self._waiting_for_connection = False

    def _connection_failed(self, link_uri, msg):
        #print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False
    
    def _connection_lost(self, link_uri, msg):
        print('Connection to %s lost: %s' % (link_uri, msg))
        self.is_connected = False
        self._waiting_for_connection = False

    def _on_key_pressed(self, key):
        if 'char' in dir(key):
            if key.char == "w":
                self._mc.start_forward(0.4)
            elif key.char  == "s":
                self._mc.start_back(0.4)
            elif key.char  == "a":
                self._mc.start_left(0.4)
            elif key.char == "d":
                self._mc.start_right(0.4)
            elif key.char == "q":
                self._mc.start_turn_left(40)
            elif key.char == "e":
                self._mc.start_turn_right(40)
        elif key == keyboard.Key.space:
            self._handle_spacebar()
        elif key == keyboard.Key.esc:
            if self.in_air:
                self._mc.land()
            self._cf.close_link()
            self.is_connected = False

    def _on_key_released(self, key):
        if 'char' in dir(key):
            if key in ["w", "s", "a", "d", "q", "e"]:
                self._mc.stop()

    def _handle_spacebar(self):
        if self.in_air:
            self._mc.land()
            self.in_air = False
        else:
            self._mc.take_off(0.5)
            self.in_air = True                
    

if __name__ == "__main__":
    # Load crazyflie driver
    cflib.crtp.init_drivers(enable_debug_driver=False)

    # Create a KeyboardController object
    controller = KeyboardController()

    str_connect = "Connecting."
    while controller._waiting_for_connection:
        print(str_connect, end="\r")
        str_connect += "."
        time.sleep(0.5)

    while controller.is_connected:
        print("IN AIR" if controller.in_air else "ON GROUND", end="\r")
        time.sleep(1)

    
