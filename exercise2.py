
from util.run_closed_loop import run_multiple, run_single
from simulation_parameters import SimulationParameters
import os
import numpy as np
import farms_pylog as pylog
import matplotlib.pyplot as plt
from wave_controller import WaveController
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
from util.rw import load_object
import itertools

def exercise2(**kwargs):

    pylog.info("Ex 2")
    pylog.info("Implement exercise 2")
    log_path = './logs/exercise2/'
    os.makedirs(log_path, exist_ok=True)

    nsim=10
    # Define parameter ranges to test
    amp_range = np.linspace(0.25, 2, nsim)
    wavefrequency_range = np.linspace(1., 3., nsim)
    epsilon_range = np.linspace(0.25, 2, nsim)
    steepness_range = np.linspace(5, 50, nsim)

    # Create a list of SimulationParameters objects for each combination of parameters
    pars_list = [
        SimulationParameters(
            simulation_i=i,
            controller="square",
            n_iterations=3001,
            log_path=log_path,
            compute_metrics=3,
            amplitude=amp,
            wavefrequency=wavefrequency,
            epsilon=epsilon,
            steepness=steepness,
            headless=True,
            print_metrics=False
        )
        for i, (amp, wavefrequency, epsilon, steepness) in enumerate(
            itertools.product(amp_range, wavefrequency_range, epsilon_range, steepness_range)
        )
    ]

    # Run all simulations
    pylog.info("Running the simulations...")
    run_multiple(pars_list)

    # Collect and plot results
    pylog.info("Collecting and plotting results...")
    results = collect_and_plot_results(log_path, len(pars_list), amp_range, epsilon_range)

    controller = load_object("logs/exercise2/controller0")

    # neural data
    state = controller.state
    metrics = controller.metrics

    # mechanical data
    links_positions = controller.links_positions  # the link positions
    links_velocities = controller.links_velocities  # the link velocities
    joints_active_torques = controller.joints_active_torques  # the joint active torques
    joints_velocities = controller.joints_velocities  # the joint velocities
    joints_positions = controller.joints_positions  # the joint positions

    left_idx = controller.muscle_l
    right_idx = controller.muscle_r
    print(controller.state.shape)
    # example plot using plot_left_right
    plot_left_right(
        controller.times,
        controller.state,
        left_idx,
        right_idx,
        cm="green",
        offset=0.1)

    # example plot using plot_trajectory
    plt.figure("trajectory")
    plot_trajectory(controller)
    plt.tight_layout()
    plt.show()
def collect_and_plot_results(logdir, n_simulations, amps,  epsilons):
    """
    Load simulation results, extract key metrics, and plot.
    """
    best_speed = -np.inf
    best_params = None
    results = np.zeros((n_simulations, 7))  # amp, freq, eps, fspeed, torque

    for i in range(n_simulations):
        file_path = f"{logdir}controller{i}"
        if not os.path.exists(file_path):
            pylog.warning(f"File not found: {file_path}")
            continue
        controller = load_object(logdir + "controller" + str(i))

        results[i] = [
            controller.pars.amplitude,
            controller.pars.epsilon,
            np.mean(controller.metrics["fspeed_cycle"]),
            np.mean(controller.metrics["lspeed_cycle"]),
            np.mean(controller.metrics["fspeed_PCA"]),
            np.mean(controller.metrics["lspeed_PCA"]),
            np.mean(controller.metrics["torque"]),

        ]
        # Assuming 'fspeed_cycle' is the forward speed metric
        forward_speed = np.mean(controller.metrics["fspeed_cycle"])

        if forward_speed > best_speed:
            best_speed = forward_speed
            best_params = {
                'simulation_id': i,
                'amplitude': controller.pars.amplitude,
                'wavefrequency': controller.pars.wavefrequency,
                'epsilon': controller.pars.epsilon,
                'steepness': controller.pars.steepness,
                'forward_speed': best_speed
            }

    # Log the best parameters
    if best_params:
        pylog.info(f"Best parameters based on forward speed: {best_params}")
    else:
        pylog.warning("No best parameters found. Check if simulations ran correctly.")


    # Convert results to a DataFrame for easier manipulation
    #results_df = pd.DataFrame(results, columns=['Amplitude', 'Frequency', 'Epsilon', 'Forward Speed', 'Torque'])
    #best_params = max(results, key=lambda item: item[2])  # item[2] is the forward speed

    #print ("Best params :", best_params)
    return results

if __name__ == '__main__':
    exercise2(headless=False)