import numpy as np
import farms_pylog as pylog

class WaveController:
    """Test controller that can switch between sine wave and enhanced square wave activations."""

    def __init__(self, pars, mode='square', steepness=25, amplitude=0.25, frequency =2.75, epsilon = 1.5):
        self.pars = pars
        self.mode = mode
        self.timestep = pars.timestep
        self.times = np.linspace(0, pars.n_iterations * pars.timestep, pars.n_iterations)
        self.n_joints = pars.n_joints
        self.state = np.zeros((pars.n_iterations, 2 * self.n_joints))
        self.muscle_l = 2 * np.arange(self.n_joints)
        self.muscle_r = self.muscle_l + 1
        pylog.warning("Controller initialized with mode: {}".format(self.mode))
        pylog.warning("Controller initialized with amplitude: {}".format(self.pars.amplitude))
        pylog.warning("Controller initialized with wavefrequency: {}".format(self.pars.wavefrequency))
        pylog.warning("Controller initialized with steepness: {}".format(self.pars.steepness))
        pylog.warning("Controller initialized with epsilon: {}".format(self.pars.epsilon))

    def sine_wave_activation(self, time, joint_index):
        A = 1.5
        epsilon = 2.
        f = 3.

        Njoints = self.pars.n_joints
        # Equation 13 from your instructions
        ml = 0.5 #+ self.pars.amplitude / 2 * np.sin(2 * np.pi * (self.pars.frequency * time
        # - self.pars.epsilon * joint_index / Njoints))
        mr = 0.5 #- self.pars.amplitude / 2 * np.sin(2 * np.pi * (self.pars.frequency* time
        #- self.pars.epsilon * joint_index / Njoints))
        return ml, mr

    def square_wave_activation(self, time, joint_index):

        phase_shift = self.pars.epsilon * joint_index / self.n_joints
        angle = 2 * np.pi * self.pars.wavefrequency * time - phase_shift
        # Use a sharper transition function
        ml = 0.5 + self.pars.amplitude/2 *np.tanh(self.pars.steepness*(np.sin(angle)))
        mr = 0.5 - self.pars.amplitude/2 *np.tanh(self.pars.steepness*(np.sin(angle)))
        return ml, mr

    def step(self, iteration, time, timestep, pos=None):
        muscles = np.zeros(2 * self.n_joints)

        for i in range(self.n_joints):
            if self.mode == 'sine':
                ml, mr = self.sine_wave_activation(time, i)
            elif self.mode == 'square':
                ml, mr = self.square_wave_activation(time, i)
            else:
                raise ValueError("Invalid mode. Choose 'sine' or 'square'.")

            muscles[2 * i] = ml
            muscles[2 * i + 1] = mr
            self.state[iteration, 2 * i] = ml
            self.state[iteration, 2 * i + 1] = mr
        return muscles