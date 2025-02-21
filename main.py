import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os


# Calculate the stimulus location in degrees of visual angle (DVA)
def get_location_dva(task_data, grid_unit, eyes_to_origin):
    # Extract target X and Y positions from the first row of task_data
    targ_x, targ_y = task_data.iloc[0][['TargPosX', 'TargPosY']]
    # Calculate the distance from the origin to the stimulus on screen
    origin_to_stim = np.sqrt((grid_unit * targ_x) ** 2 + (grid_unit * targ_y) ** 2)
    # Compute the visual angle in radians and convert to degrees
    dva_radians = 2 * np.arctan(origin_to_stim / eyes_to_origin)
    return round(np.degrees(dva_radians), 2)

# Calculate the stimulus diameter in degrees of visual angle (DVA)
def get_diameter_dva(task_data, grid_unit, eyes_to_origin):
    # Extract target X and Y positions from the first row of task_data
    targ_x, targ_y = task_data.iloc[0][['TargPosX', 'TargPosY']]
    # Compute the distances along the X and Y axes
    targ_x_dist = grid_unit * targ_x
    targ_y_dist = grid_unit * targ_y
    # Calculate the Euclidean distance from eyes to the stimulus
    eyes_to_stim = np.sqrt(eyes_to_origin ** 2 + targ_x_dist ** 2 + targ_y_dist ** 2)
    # Determine the angle phi between the stimulus and the center of the screen
    phi = np.arctan2(np.sqrt(targ_x_dist ** 2 + targ_y_dist ** 2), eyes_to_origin)
    # Calculate the effective size of the stimulus on screen considering the angle phi
    effective_size = (grid_unit * task_data.iloc[0]['TargSize']) * np.cos(phi)
    # Compute the visual angle in radians based on effective size and convert to degrees
    dva_radians = 2 * np.arctan((effective_size / 2) / eyes_to_stim)
    return round(np.degrees(dva_radians), 2)

# Process condition data to group by target orientation and spatial location
def get_condition_pack(condition):
    data, cue, rf = condition
    # Filter data based on cue condition
    cue_data = data[data["Cued"] == cue]

    # Determine the receptive field (RF) position
    rf_x, rf_y = cue_data.iloc[0][['RFposX', 'RFposY']]
    # Set condition for X based on RF x position polarity
    x_condition = cue_data['TargPosX'] > 0 if rf_x > 0 else cue_data['TargPosX'] < 0
    # Set condition for Y based on RF y position polarity
    y_condition = cue_data['TargPosY'] > 0 if rf_y > 0 else cue_data['TargPosY'] < 0
    # Invert conditions if rf flag is not set
    if not rf:
        x_condition, y_condition = ~x_condition, ~y_condition

    # Filter data based on both X and Y conditions
    loc_data = cue_data[x_condition & y_condition]
    # Create a sorted list of unique X positions based on RF condition
    x_pos_list = sorted(loc_data['TargPosX'].unique(), reverse=not rf)
    # For each X position, extract the corresponding Y position
    y_pos_list = [loc_data.loc[loc_data['TargPosX'] == x, 'TargPosY'].iloc[0] for x in x_pos_list]

    condition_pack = []
    # Group the data by target orientation for each unique location (x, y)
    for x, y in zip(x_pos_list, y_pos_list):
        masked_data = loc_data.query('TargPosX == @x and TargPosY == @y')
        condition_pack.append([group for _, group in masked_data.groupby('TargOri', sort=True)])

    return condition_pack

# Generate a list of orientation labels for each group in a location
def get_ori_labels(location):
    return [int(group['TargOri'].iloc[0]) for group in location]

# Calculate firing rates for a specific location based on spike times and a given time window
def get_location_rates(spike_data, window, location):
    # Convert spike times from the spike_data to a numpy array of type float32
    spikes_times = np.asarray(spike_data['spike_times'], dtype=np.float32)

    location_rates = []
    # Loop through each orientation group in the location
    for ori_group in location:
        # Get reference times for the group and compute end times based on the window
        ref_times = np.array(ori_group['ref_times'])[:, None]
        end_times = ref_times + window
        # Count spikes that occur within the time window for each reference time
        spike_count = ((spikes_times >= ref_times) & (spikes_times <= end_times)).sum(axis=1)
        # Compute firing rate by dividing spike count by the window duration
        location_rates.append(spike_count / window)

    return location_rates

# Define a Gaussian function for curve fitting
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Fit a Gaussian curve to the provided location data (orientation vs. firing rate)
def fit_gaussian_curve(location):
    x_data = location[0]
    y_data = location[1]
    try:
        # Provide an initial guess for amplitude, mean, and standard deviation
        initial_guess = [np.max(y_data), np.mean(x_data), np.std(x_data)]
        curve_params, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    except:
        curve_params = None
    return curve_params

# Calculate the polar angle (in degrees) for a location based on target positions
def get_location_degree(location):
    location = pd.concat(location)
    targ_x, targ_y = location.iloc[0][['TargPosX', 'TargPosY']]
    # Compute the angle using arctan2 for correct quadrant determination
    loc_radians = np.arctan2(targ_y, targ_x)
    return np.degrees(loc_radians) % 360

# Streamlit application entry point 
def app():
    # Set up the Streamlit page configuration and title
    st.set_page_config(page_title="FineEffect")
    st.title("FineEffect")

    # Input field for the TDT folder path
    tdt_path = st.text_input("TDT folder path:")
    st.success(f'{os.cpu_count()} CPU cores available')

    # Configuration panel for visual angle parameters
    with st.expander('Configue DVA parameters'):
        eyes_to_origin = float(st.text_input('Distance from eyes to center of screen (cm):', value='64.77'))
        grid_unit = float(st.text_input('Length of on-screen grid unit (cm):', value='1.06'))

    # Input field for the effect window duration in seconds
    window = float(st.text_input('Effect window (sec):', value=1))

    # Proceed if a valid TDT path is provided
    if tdt_path:
        # Load task and spike data from CSV files
        task_data = pd.read_csv(tdt_path + "/TimeLock.csv")
        spike_data = pd.read_csv(tdt_path + "/SortView.csv")

        # Compute the stimulus location and diameter in DVA
        location_dva = get_location_dva(task_data, grid_unit, eyes_to_origin)
        diameter_dva = get_diameter_dva(task_data, grid_unit, eyes_to_origin)

        st.write(f'Stimuli locations: {location_dva} DVA')
        st.write(f'Stimuli diameters: {diameter_dva} DVA')

        # Define conditions: cued RF, cued away, and uncued RF
        conditions = [[task_data, 1, 1], [task_data, 1, 0], [task_data, 0, 1]]

        # Process conditions in parallel to obtain condition packs
        with ProcessPoolExecutor() as executor:
            condition_pack = list(executor.map(get_condition_pack, conditions))

        # Extract orientation labels for each condition pack in parallel
        ori_pack = []
        for pack_locations in condition_pack:
            with ProcessPoolExecutor() as executor:
                condition_labels = list(executor.map(get_ori_labels, pack_locations))
                ori_pack.append(condition_labels)

        # Calculate firing rates for each location within each condition pack in parallel
        rate_pack = []
        for pack_locations in condition_pack:
            with ProcessPoolExecutor() as executor:
                func = partial(get_location_rates, spike_data, window)
                condition_rates = list(executor.map(func, pack_locations))
                rate_pack.append(condition_rates)

        # Normalize firing rates by the maximum rate found across all conditions and locations
        max_rate = np.max([i for con in rate_pack for loc in con for ori in loc for i in ori])
        rate_pack = [[[ori / max_rate for ori in loc] for loc in con] for con in rate_pack]

        # Prepare orientation tuning data by averaging firing rates for each orientation
        ori_tunings = []
        for i1 in range(len(rate_pack)):
            locations = ori_pack[i1]
            condition = []
            for i2 in range(len(locations)):
                ori_list = ori_pack[i1][i2]
                rates = [np.mean(r) for r in rate_pack[i1][i2]]
                condition.append([ori_list, rates])
            ori_tunings.append(condition)

        # Fit Gaussian curves to the orientation tuning data in parallel
        ori_curves= []
        for pack_locations in ori_tunings:
            with ProcessPoolExecutor() as executor:
                condition_curves = list(executor.map(fit_gaussian_curve, pack_locations))
                ori_curves.append(condition_curves)

        # Compute the polar degrees of locations for the first condition pack in parallel
        with ProcessPoolExecutor() as executor:
            location_degrees = list(executor.map(get_location_degree, condition_pack[0]))

        # Prepare spatial tuning data: mean firing rates and standard deviations for each location
        space_tuning = []
        for i1 in range(len(rate_pack)):
            rate_means = []
            rate_stds = []
            for i2 in range(len(location_degrees)):
                rates = [np.mean(r) for r in rate_pack[i1][i2]]
                rate_means.append(np.mean(rates))
                rate_stds.append(np.std(rates))
            space_tuning.append([location_degrees, rate_means, rate_stds])

        # Fit Gaussian curves to the spatial tuning data in parallel
        with ProcessPoolExecutor() as executor:
            space_curves = list(executor.map(fit_gaussian_curve, space_tuning))

        # Generate a range of X values for plotting the fitted curves
        x_range = np.linspace(location_degrees[0], location_degrees[-1], 100)

        # Plot the spatial tuning and fitted Gaussian for the cued RF condition
        plt.errorbar(
            space_tuning[0][0],
            space_tuning[0][1],
            xerr=space_tuning[0][2],
            yerr=space_tuning[0][2],
            fmt='o',
            color='black',
            capsize=3,
            capthick=1,
            label='cued RF'
        )
        amp, mu, sigma = space_curves[0]
        plt.plot(
            x_range,
            gaussian(x_range, amp, mu, sigma),
            color='black',
            linestyle='--'
        )

        # Plot the spatial tuning and fitted Gaussian for the cued away condition
        plt.errorbar(
            space_tuning[1][0],
            space_tuning[1][1],
            xerr=space_tuning[1][2],
            yerr=space_tuning[1][2],
            fmt='o',
            color='red',
            capsize=3,
            capthick=1,
            label='cued away'
        )
        amp, mu, sigma = space_curves[1]
        plt.plot(
            x_range,
            gaussian(x_range, amp, mu, sigma),
            color='red',
            linestyle='--'
        )

        # Plot the spatial tuning and fitted Gaussian for the uncued RF condition
        plt.errorbar(
            space_tuning[2][0],
            space_tuning[2][1],
            xerr=space_tuning[2][2],
            yerr=space_tuning[2][2],
            fmt='o',
            color='gray',
            capsize=3,
            capthick=1,
            label='uncued RF'
        )
        amp, mu, sigma = space_curves[2]
        plt.plot(
            x_range,
            gaussian(x_range, amp, mu, sigma),
            color='gray',
            linestyle='--'
        )

        # Label the plot axes and add a legend
        plt.xlabel('Location (degrees)')
        plt.ylabel('Normalized firing rate')
        plt.legend()

        # Render the plot in the Streamlit app and close the plot to free memory
        st.pyplot(plt)
        plt.close()

# Run the app if the script is executed directly
if __name__ == '__main__':
    app()
