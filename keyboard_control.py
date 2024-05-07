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
from cflib.crazyflie.log import LogConfig


URI = uri_helper.uri_from_env(default="radio://0/40/2M/E7E7E7E704")

BASE_VEL = 0.3
DESIRED_HEIGHT = 0.4
BASE_ANG_RATE = 30

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

        self._command = [0,0,0,0]

    def _connected(self, link_uri):
        print('Connected to %s' % link_uri)
        self._waiting_for_connection = False
        self.is_connected = True

        # Create a MotionCommander object
        self._mc = MotionCommander(self._cf)

        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')

        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

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

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        #print(f'[{timestamp}][{logconf.name}]: ', end='\r')
        for name, value in data.items():
            print(f'{name}: {value:3.3f} ', end='')
        print('', end='\r')

    def _on_key_pressed(self, key):
        if 'char' in dir(key):
            if key.char == "w":
                self._command = [BASE_VEL,0,0,DESIRED_HEIGHT]
            elif key.char  == "s":
                self._command = [-BASE_VEL,0,0,DESIRED_HEIGHT]
            elif key.char  == "a":
                self._command = [0,BASE_VEL,0,DESIRED_HEIGHT]
            elif key.char == "d":
                self._command = [0,-BASE_VEL,0,DESIRED_HEIGHT]
            elif key.char == "q":
                self._command = [0,0,BASE_ANG_RATE,DESIRED_HEIGHT]
            elif key.char == "e":
                self._command = [0,0,-BASE_ANG_RATE,DESIRED_HEIGHT]
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
                self._command = [0,0,0,DESIRED_HEIGHT]

    def _handle_spacebar(self):
        if self.in_air:
            self.in_air = False
            # for y in range(10):
            #     self._cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
            #     time.sleep(0.1)
            self._mc.land()
        else:
            # for y in range(10):
            #     self._cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
            #     time.sleep(0.1)
            self._mc.take_off(DESIRED_HEIGHT)
            self._command = [0,0,0,DESIRED_HEIGHT]
            self.in_air = True                
    

if __name__ == "__main__":
    # Load crazyflie driver
    cflib.crtp.init_drivers(enable_debug_driver=False)

    # Create a KeyboardController object
    controller = KeyboardController()

    cf = controller._cf

    str_connect = "Connecting."
    while controller._waiting_for_connection:
        print(str_connect, end="\r")
        str_connect += "."
        time.sleep(0.5)

    print("INITIALIZING KALMAN ESTIMATOR", end="\r")
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    print("INITIALIZING KALMAN ESTIMATOR: DONE")

    while controller.is_connected:
        #print("IN AIR" if controller.in_air else "ON GROUND", end="\r")
        if controller.in_air:
            cf.commander.send_hover_setpoint(*controller._command)
        time.sleep(0.1)

    
