import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')
import numpy as np



from matplotlib.widgets import Button, Slider

from cycler import cycler
import pandas as pd
from pypet import Trajectory
from mag_vs_residual_graph_traj import WFS_SIMU_VISU

FILENAME = "/mnt/home/usager/cars2019/Documents/Programming/OOPAO_TRWFS/trwfs/sum/res/20m_3_5mod_increased_subap_12feb2024.hdf5"


results_to_display = ["SR_H", "Turbulence", "Residual", 'wfsSignal', 'OPD', 'a_est']

WSV = WFS_SIMU_VISU(FILENAME)

explored_parameters_df = WSV.get_explored_parameters_and_combinations()

parameters = list(explored_parameters_df.keys())
parameters_exception = "parameters.ao_calib_file"
parameters.remove(parameters_exception)



def fetch_data():
    to_plot = WSV.get_graph(y=results_to_display,
                            remove_first=0,
                            parameters=parameters)

    selected_parameter_values = {el: WSV.traj[el] for el in parameters}

    selected_parameter_values["parameters.enable_custom_frames"] = True
    selected_parameter_values["parameters.magnitude"] = 15.0
    selected_parameter_values["parameters.gainCL"] = 0.2



    mask = pd.DataFrame([to_plot[key] == val for key, val in selected_parameter_values.items()]).T.all(axis=1)
    to_display = to_plot[mask]

    return to_display

to_display = fetch_data()


fig, axs = plt.subplots(2, 3)
SR_H_line, = axs[0, 0].plot(to_display[results_to_display[0]].iloc[0])

axs[0, 0].set_title("Strehl Ratio (H band)")
turbulence_line,  = axs[0, 1].plot(to_display[results_to_display[1]].iloc[0], label="Turbulence")
residual_line,  = axs[0, 1].plot(to_display[results_to_display[2]].iloc[0], label="Residual")
axs[0, 1].set_title("Turbulence and residual")
axs[0, 1].legend()

wfs_signal_im  = axs[1, 0].imshow(to_display[results_to_display[3]].iloc[0][-1,:,:])
wfs_signal_im_cb = plt.colorbar(wfs_signal_im, ax=axs[1, 0])
axs[1, 0].set_title("WFS image")

OPD_im = axs[1, 1].imshow(to_display[results_to_display[4]].iloc[0][-1,:,:])
OPD_im_cb = plt.colorbar(OPD_im, ax=axs[1, 1])
axs[1, 1].set_title("Telescope OPD")

max_modes = len(to_display[results_to_display[5]].iloc[0][-1,:])
a_est_x = list(range(max_modes))
a_est_bar = axs[1, 2].bar(a_est_x, to_display[results_to_display[5]].iloc[0][-1,:])
axs[1, 2].set_title("KL modes estimations")


#ax = to_plot.plot(kind='line', grid=True, xlabel=xlabel, ylabel=ylabel, ax=ax, logx=logx, logy=logy)  # , marker='o')


axloop = fig.add_axes([0.1, 0.01, 0.65, 0.03])
loop_slider = Slider(
    ax=axloop,
    label='Loop Iteration',
    valmin=0,
    valmax=len(to_display[results_to_display[0]].iloc[0])-1,
    valstep=list(range(len(to_display[results_to_display[0]].iloc[0]))),
    valinit=len(to_display[results_to_display[0]].iloc[0])-1,
)

# The function to be called anytime a slider's value changes
def update_position(val):
    #to_display = fetch_data()
    data = np.copy(to_display[results_to_display[0]].iloc[0])
    data[val:] = np.nan
    SR_H_line.set_ydata(data)

    data = np.copy(to_display[results_to_display[1]].iloc[0])
    data[val:] = np.nan
    turbulence_line.set_ydata(data)

    data = np.copy(to_display[results_to_display[2]].iloc[0])
    data[val:] = np.nan
    residual_line.set_ydata(data)

    data = np.copy(to_display[results_to_display[3]].iloc[0][val,:,:])
    wfs_signal_im.set_data(data)
    wfs_signal_im.set_clim(vmin=np.min(data), vmax=np.max(data))

    data = np.copy(to_display[results_to_display[4]].iloc[0][val, :, :])
    OPD_im.set_data(data)
    OPD_im.set_clim(vmin=np.min(data), vmax=np.max(data))


    data = np.copy(to_display[results_to_display[5]].iloc[0][val, :])
    axs[1, 2].set_ylim([np.min(data), np.max(data)])
    for rect, h in zip(a_est_bar, data):
        rect.set_height(h)


    fig.canvas.draw_idle()

loop_slider.on_changed(update_position)

plt.show()
### Display figure settings