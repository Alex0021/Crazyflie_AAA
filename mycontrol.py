from occupancyMap import OccupancyMap
import numpy as np

occupancy_map = OccupancyMap()

STARTING_POS = [1.5, 0.5]


def get_command(sensor_data, dt):
    sensor_data['x_global'] = sensor_data['x_global'] + STARTING_POS[0]
    sensor_data['y_global'] = sensor_data['y_global'] + STARTING_POS[1]
    occupancy_map.update(sensor_data, np.array([1,1]), np.array([0,1]), np.array([1,0]))