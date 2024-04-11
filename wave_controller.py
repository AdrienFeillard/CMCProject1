# wave_controller.py
import numpy as np
import farms_pylog as pylog

class WaveController:
    """Test controller"""
    def __init__(self, pars):
        self.pars = pars
        self.timestep = pars.timestep
        # Additional properties for model simulation
        self.times = np.linspace(0, pars.n_iterations * pars.timestep, pars.n_iterations)
        self.n_joints = pars.n_joints
        self.state = np.zeros((pars.n_iterations, 2 * self.n_joints))
        self.muscle_l = 2 * np.arange(self.n_joints)
        self.muscle_r = self.muscle_l + 1
        self.links_positions = np.zeros((pars.n_iterations, pars.n_joints, 3))
        pylog.warning("Implement the step function following the instructions here and in the report")

    def sine_wave_activation(self, time, joint_index):
        phase_shift = self.pars.phase * joint_index / self.n_joints
        ml = 0.5 + self.pars.amplitude/2 * np.sin(2 * np.pi * self.pars.frequency * time - phase_shift)
        mr = 0.5 - self.pars.amplitude/2 * np.sin(2 * np.pi * self.pars.frequency * time - phase_shift)
        return ml, mr

    def step(self, iteration, time, timestep, pos=None):
        activations = np.zeros(2 * self.n_joints)
        for i in range(self.n_joints):
            ml, mr = self.sine_wave_activation(time, i)
            activations[self.muscle_l[i]] = ml
            activations[self.muscle_r[i]] = mr
        self.state[iteration, :] = activations
        self.links_positions[iteration] = self.sim.data.qpos.reshape(-1, 3)
        return activations

