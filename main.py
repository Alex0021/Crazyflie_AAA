import logging
import sys
import time
from threading import Event
from pynput import keyboard
from mycontrol import get_command
from occupancyMap import OccupancyMap

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig


URI = uri_helper.uri_from_env(default="radio://0/40/2M/E7E7E7E704")

class MyController:
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

        # Initialize the occupancy map
        self._occupancy_map = OccupancyMap()

    def _connected(self, link_uri):
        print('Connected to %s' % link_uri)
        self._waiting_for_connection = False
        self.is_connected = True

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')
        self._lg_stab.add_variable('range.front')
        self._lg_stab.add_variable('range.back')
        self._lg_stab.add_variable('range.left')
        self._lg_stab.add_variable('range.right')
        self._lg_stab.add_variable('range.down')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
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

        # Create a MotionCommander object
        self._mc = MotionCommander(self._cf)

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        # print(f'[{timestamp}][{logconf.name}]: ', end='')
        # for name, value in data.items():
        #     print(f'{name}: {value:3.3f} ', end='')
        # print()
        sensor_data = {
            'x_global': data['stateEstimate.x'],
            'y_global': data['stateEstimate.y'],
            'z_global': data['stateEstimate.z'],
            'yaw': data['stabilizer.yaw'],
            'range_down': data['range.down'],
            'range_front': data['range.front'],
            'range_back': data['range.back'],
            'range_left': data['range.left'],
            'range_right': data['range.right']
        }
        next_cmd = get_command(sensor_data)

    def _disconnected(self, link_uri):
        print('Disconnected from %s' % link_uri)
        self.is_connected = False
        self._waiting_for_connection = False

    def _connection_failed(self, link_uri, msg):
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False
        self._waiting_for_connection = False
    
    def _connection_lost(self, link_uri, msg):
        print('Connection to %s lost: %s' % (link_uri, msg))
        self.is_connected = False
        self._waiting_for_connection = False

if __name__ == "__main__":
    print('==============| TEAM 4: AAA |===============')

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    print("Initializing driver: COMPLETED")
    
    # Instanciate the controller
    controller = MyController()

    cf = controller._cf

    print("INITIALIZING KALMAN ESTIMATOR", end="\r")
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    print("INITIALIZING KALMAN ESTIMATOR: DONE")
