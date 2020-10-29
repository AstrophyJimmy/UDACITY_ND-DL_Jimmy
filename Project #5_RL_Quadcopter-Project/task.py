import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
                                    init_angle_velocities=None, runtime=5., target_pos=None):
        
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        penality = 0
        currentPos = self.sim.pose[:3]
        currentAngles = self.sim.pose[3:]
        targetPos = self.target_pos

# Penalizing for being far from object while increasing the penality in case of z Axis by an order of 10 as it is the ultimate goal
        dxSq = (currentPos[0]-targetPos[0])**2
        dySq = (currentPos[1]-targetPos[1])**2
        dzSq = (currentPos[2]-targetPos[2])**2
        
        penality +=  dxSq + dySq + 10*dzSq
        
# linking velocity to residual distance
        penality += abs(sum(abs(currentPos-targetPos)) - sum(abs(self.sim.v)))
        
# Penalizing for not unneeded rotations as they result in extra torque or unnecessary motion to overcome 
# As I am considering 0 for angles for stability
        penality += sum(currentAngles)

        distFromTarget = np.sqrt(dxSq + dySq + dzSq)

# Rewarding for being in the neigborhod for the target
        if distFromTarget < 1:
            reward += 1000
        
# Fixed Reward for staying flying
        reward += 100
        
        
        return reward - penality*0.0002


    def current_state(self):
        """to get information about current position, velocity and angular velocity""" 
        return np.concatenate([np.array(self.sim.pose), np.array(self.sim.v), np.array(self.sim.angular_v)])  

    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        #print(np.shape(reward))
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
    
    
    