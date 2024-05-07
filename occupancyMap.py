import numpy as np
import matplotlib.pyplot as plt

class OccupancyMap():

    def __init__(self):
        # Occupancy map based on distance sensor
        self.min_x, self.max_x = 0, 5.0 # meter
        self.min_y, self.max_y = 0, 3.0 # meter
        self.range_max = 2.0 # meter, maximum range of distance sensor
        self.res_pos = 0.1 # meter
        self.conf = 0.2 # certainty given by each measurement
        self.t = 0 # only for plotting TODO: Can we use this in the hardware version? This needs to be passed in.

        self.map = np.zeros((int((self.max_x-self.min_x)/self.res_pos), int((self.max_y-self.min_y)/self.res_pos))) # 0 = unknown, 1 = free, -1 = occupied

    def update(self, sensor_data):
        pos_x = sensor_data['x_global']
        pos_y = sensor_data['y_global']
        yaw = sensor_data['yaw']
        
        for j in range(4): # 4 sensors
            yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
            if j == 0:
                measurement = sensor_data['range_front']
            elif j == 1:
                measurement = sensor_data['range_left']
            elif j == 2:
                measurement = sensor_data['range_back']
            elif j == 3:
                measurement = sensor_data['range_right']
            
            for i in range(int(self.range_max/self.res_pos)): # range is 2 meters
                dist = i*self.res_pos
                idx_x = int(np.round((pos_x - self.min_x + dist*np.cos(yaw_sensor))/self.res_pos,0))
                idx_y = int(np.round((pos_y - self.min_y + dist*np.sin(yaw_sensor))/self.res_pos,0))

                # make sure the current_setpoint is within the map
                if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > self.range_max:
                    break

                # update the map
                if dist < measurement:
                    map[idx_x, idx_y] += self.conf
                else:
                    map[idx_x, idx_y] -= self.conf
                    break
        
        map = np.clip(map, -1, 1) # certainty can never be more than 100%

        # Add a perimeter of obstacles around the occupancy map
        map = self.add_perimeter_obstacles(map)

        # only plot every Nth time step (comment out if not needed) 
        # TODO: Maybe dont use self.t
        if self.t % 50 == 0:
            plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
            plt.savefig("map.png")
            plt.close()
        return map
    
    # Add a boarder of obstacles around the perimeter of the occupancy map
    def add_perimeter_obstacles(self, map):
        '''
        This function adds a perimeter of obstacles around the occupancy map
        '''
        map[0, :] = -1
        map[-1, :] = -1
        map[:, 0] = -1
        map[:, -1] = -1
        return map