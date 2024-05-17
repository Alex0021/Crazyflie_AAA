import logging
import sys
import time
from threading import Event
from pynput import keyboard
from mycontrol import get_command
import math
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig


URI = uri_helper.uri_from_env(default="radio://0/40/2M/E7E7E7E704")

BASE_VEL = 0.3
DESIRED_HEIGHT = 0.5
BASE_ANG_RATE = 30
EMERGENCY_LANDING_UP_RANGE = 0.1

class KeyboardController:
    def __init__(self):
        # Create a Crazyflie object
        self._cf = Crazyflie(rw_cache='./cache')

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self.in_air = False
        self.is_ready = False
        self.force_landing = False
        self.print_data = 0

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
        self._lg_stab.add_variable('range.front')
        self._lg_stab.add_variable('range.back')
        self._lg_stab.add_variable('range.left')
        self._lg_stab.add_variable('range.right')
        self._lg_stab.add_variable('range.zrange')
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
        listener = keyboard.Listener(on_press=self._on_key_pressed)
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
        if not self.is_ready or self.force_landing:
            return
        # if data['range.up'] < EMERGENCY_LANDING_UP_RANGE:
        #     self._handle_spacebar()
        #     return
        self.sensor_data = {
            'x_global': data['stateEstimate.x'],
            'y_global': data['stateEstimate.y'],
            'z_global': data['stateEstimate.z'],
            'yaw': data['stabilizer.yaw']/180*math.pi,
            'range_down': data['range.zrange']/1000,
            'range_front': data['range.front']/1000,
            'range_back': data['range.back']/1000,
            'range_left': data['range.left']/1000,
            'range_right': data['range.right']/1000
        }
        #print(f"Z Global: {self.sensor_data['z_global']:.3f}")
        #print("Range down: ", self.sensor_data['range_down'])
        # if self.print_data % 1 == 0:
        #     for name, value in self.sensor_data.items():
        #         print(f'{name}: {value:3.3f} ', end='\n')
        cmd = get_command(self.sensor_data)
        self._command = [cmd[0],cmd[1],cmd[3],cmd[2]]
        self.print_data += 1


    def _on_key_pressed(self, key):
        if not self.is_ready:
            return
        if key == keyboard.Key.space:
            self._force_landing()
        elif key == keyboard.Key.esc:
            self.is_ready = False
            self._cf.commander.send_stop_setpoint()
            self._cf.close_link()
            self.is_connected = False

    def _force_landing(self):    
        self.force_landing = True
        for y in range(10):
            self._command = [0, 0, 0, (10 - y) / 20]
            time.sleep(0.1)
        time.sleep(0.5)
        self.is_ready = False
        self._cf.commander.send_stop_setpoint()
        self._cf.close_link()

    def _handle_spacebar(self):
        if self.in_air or self.sensor_data['range_down'] > 0.2:
            self.in_air = False
            
            self.is_ready = False
            print("LANDING...")

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
    time.sleep(0.5)
    print("INITIALIZING KALMAN ESTIMATOR: DONE")
    controller.is_ready = True

    plt.show(block=False)

    while controller.is_connected:
        #print("IN AIR" if controller.in_air else "ON GROUND", end="\r")
        #if controller.in_air:
        if controller.is_ready:
            cf.commander.send_hover_setpoint(*controller._command)
        time.sleep(0.01)

    
