import numpy as np
import matplotlib.pyplot as plt
import cv2

class OccupancyMap():

    def __init__(self):
        # Occupancy map based on distance sensor
        self.min_x, self.max_x = 0, 5.0 # meter
        self.min_y, self.max_y = 0, 3.0 # meter
        self.range_max = 2.0 # meter, maximum range of distance sensor
        self.res_pos = 0.1 # meter
        self.conf = 0.2 # certainty given by each measurement
        self.t = 0 # only for plotting TODO: Can we use this in the hardware version? This needs to be passed in.
        self.map_size_x = self.max_x / self.res_pos # blocks
        self.map_size_y = self.max_y / self.res_pos # blocks
        self.goal = None # current goal position

        self.map = np.zeros((int((self.max_x-self.min_x)/self.res_pos), int((self.max_y-self.min_y)/self.res_pos))) # 0 = unknown, 1 = free, -1 = occupied
        self.canvas = np.zeros((self.map_size_y, self.map_size_x, 3), dtype=np.uint8) * 255  # Swap canvas dimensions

    def update(self, sensor_data, goal, attractive_force, repulsive_force):
        pos_x = sensor_data['x_global']
        pos_y = sensor_data['y_global']
        yaw = sensor_data['yaw']
        self.goal = goal
        
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
        # if self.t % 50 == 0:
        #     plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        #     plt.savefig("map.png")
        #     plt.close()
        # self.t+=1

        self.update_visualization(sensor_data, map, attractive_force, repulsive_force)
        return map
    
    def update_visualization(self, sensor_data, map, attractive_force, repulsive_force):

        arrow_size = 10
        map_size_x = 3 / self.res_pos 
        map_size_y = 5 / self.res_pos
        
        # TODO: If we pass these values in they act as gains for adjusting rhe attactive and repulsive forces
        k_a = 1
        k_r = 1
        
        # Calculate Resultant Force in World Frame for Visualization
        resultant_force = (k_a*attractive_force) + (k_r*repulsive_force)
        
        if self.t % 50 == 0:
            # Swap X and Y axes for visualization
            xdrone = int(sensor_data['y_global'] / self.res_pos)  # Swap X and Y axes
            ydrone = int(sensor_data['x_global'] / self.res_pos)  # Swap X and Y axes
            xgoal = int(self.goal[1] / self.res_pos)  # Swap X and Y axes
            ygoal = int(self.goal[0] / self.res_pos)  # Swap X and Y axes
            
            # Clear canvas
            self.canvas.fill(255)
            
            # Plot the map with upscaling (Comment out if maps are the same size)
            map = np.kron(map, np.ones((10, 10)))
            idx_obstacles = np.where(map < 0)

            self.canvas[map_size_y-idx_obstacles[0]-1, map_size_x-idx_obstacles[1]-1] = (0, 0, 255)  # Red
            
            # Plot drone and goal positions
            cv2.circle(self.canvas, (map_size_x - xdrone, map_size_y - ydrone), 5, (0, 0, 255), -1)  # Red for drone, mirror X coordinate
            cv2.circle(self.canvas, (map_size_x - xgoal, map_size_y - ygoal), 5, (255, 0, 0), -1)  # Blue for goal, mirror X coordinate

            # TODO: If we want to use a bunch of different goal locations for grid search, we can use this code for visualization
            # Plot the possible landing pad locations
            # if num_possible_pads_locations is not None and num_possible_pads_locations > 0:
            #     for location in possible_pad_locations:
            #         cv2.circle(canvas, (map_size_x - int(location[1] * 10), map_size_y - int(location[0] * 10)), 5, (0, 255, 0), -1)

            # Plot the attractive force vector
            attractive_magnitude = np.linalg.norm(attractive_force)
            if attractive_magnitude != 0:
                arrow_end_point = (map_size_x - (xdrone + int(attractive_force[1] * arrow_size)), map_size_y - (ydrone + int(attractive_force[0] * arrow_size)))
                cv2.arrowedLine(self.canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (0, 255, 0), thickness=1, tipLength=0.3)

            # Plot the repulsive force vector
            repulsive_magnitude = np.linalg.norm(repulsive_force)
            if repulsive_magnitude != 0:
                arrow_end_point = (map_size_x - (xdrone + int(repulsive_force[1] * arrow_size)), map_size_y - (ydrone + int(repulsive_force[0] * arrow_size)))
                cv2.arrowedLine(self.canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (255, 0, 0), thickness=1, tipLength=0.3)

            # Plot the resultant force vector
            resultant_magnitude = np.linalg.norm(resultant_force)
            if resultant_magnitude != 0:
                arrow_end_point = (map_size_x - (xdrone + int(resultant_force[1] * arrow_size)), map_size_y - (ydrone + int(resultant_force[0] * arrow_size)))
                cv2.arrowedLine(self.canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (0, 0, 0), thickness=1, tipLength=0.3)
            
            # Show the updated canvas
            cv2.imshow("Map", self.canvas)
            cv2.waitKey(1)  # Wait for a short time to update the display
    
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