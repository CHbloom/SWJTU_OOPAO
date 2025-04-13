from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from cycler import cycler
import pandas as pd
from pypet import Trajectory
from mag_vs_residual_graph_traj import WFS_SIMU_VISU

FILENAME = "/mnt/home/usager/cars2019/Documents/Programming/OOPAO_TRWFS/trwfs/sum/res/with_old_param_file_long_25jan2024.hdf5"


results_to_display = ["SR_H", "Turbulence", "Residual", 'wfsSignal', 'OPD', 'a_est', 'photons_per_subArea']

WSV = WFS_SIMU_VISU(FILENAME)

explored_parameters_df = WSV.get_explored_parameters_and_combinations()

parameters = list(explored_parameters_df.keys())
parameters_exception = "parameters.ao_calib_file"
parameters.remove(parameters_exception)
parameters_additions = ['parameters.r0', 'parameters.lightThreshold', 'parameters.magnitude', 'parameters.gainCL', 'parameters.enable_custom_frames']
parameters.extend(parameters_additions)
parameters = list(dict.fromkeys(parameters))  # To remove duplicates from the list

# Load default values from the traj of file
selected_parameter_values = {el: WSV.traj[el] for el in parameters}


t = WSV.fetch_plot_data(x="photons_per_subArea", y="SR_H", selected_parameter_values=selected_parameter_values)


def fetch_data(selected_parameter_values=None):


    to_plot = WSV.get_graph(y=results_to_display,
                            remove_first=0,
                            parameters=parameters + ['parameters.r0', 'parameters.lightThreshold'])

    if selected_parameter_values is None:
        selected_parameter_values = {el: WSV.traj[el] for el in parameters}

    selected_parameter_values["parameters.enable_custom_frames"] = True
    selected_parameter_values["parameters.gainCL"] = 0.4
    selected_parameter_values["parameters.magnitude"] = 13.0



    title = f"Mag={to_display['parameters.magnitude'].iloc[0]}, " \
            f"Photon_per_sub={to_display['photons_per_subArea'].iloc[0]}, " \
            f"TR={to_display['parameters.enable_custom_frames'].iloc[0]}, " \
            f"R0={to_display['parameters.r0'].iloc[0]}, " \
            f"LT={to_display['parameters.lightThreshold'].iloc[0]}" \
            f"gainCL={to_display['parameters.gainCL'].iloc[0]}"


    return to_display, selected_parameter_values, title

to_display, current_parameter_values, title_to_use = fetch_data()




import plotly.express as px



N = 1500

fig = make_subplots(
    rows=2, cols=2, subplot_titles=(title_to_use, ""),
    horizontal_spacing=0.051
)

fig.add_trace(go.Line(y=to_display[results_to_display[0]].iloc[0]), row=1, col=1)
fig.add_trace(go.Line(y=to_display[results_to_display[1]].iloc[0]), row=1, col=2)
fig.add_trace(go.Line(y=to_display[results_to_display[2]].iloc[0]), row=1, col=2)
fig.add_trace(go.Heatmap(z=to_display[results_to_display[3]].iloc[0][N, :, :]), row=2, col=1)
fig.add_trace(go.Heatmap(z=to_display[results_to_display[4]].iloc[0][N, :, :]), row=2, col=2)

frames = [
    go.Frame(data=[go.Line(y=to_display[results_to_display[0]].iloc[0][:i]),
                   go.Line(y=to_display[results_to_display[1]].iloc[0][:i]),
                   go.Line(y=to_display[results_to_display[2]].iloc[0][:i]),
                   go.Heatmap(z=to_display[results_to_display[3]].iloc[0][i, :, :]),
                   go.Heatmap(z=to_display[results_to_display[4]].iloc[0][i, :, :])
                   ], name=i,
             traces=[0,1,2,3,4])
    for i in range(1, N)
]

fig.frames = frames

fig.update_layout(
    updatemenus=[
        {
            "buttons": [{"args": [None, {"frame": {"duration": 500, "redraw": True}}],
                         "label": "Play", "method": "animate",},
                        {"args": [[None],{"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate", "transition": {"duration": 0},},],
                         "label": "Pause", "method": "animate",},],
            "type": "buttons",
        }
    ],
    # iterate over frames to generate steps... NB frame name...
    sliders=[{"steps": [{"args": [[f.name],{"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",},],
                         "label": f.name, "method": "animate",}
                        for f in frames],}],
    height=1200,
    width=1600
    #yaxis={"title": 'callers'},
    #xaxis={"title": 'callees', "tickangle": 45, 'side': 'top'},
    #title_x=0.5

)

fig.show()


#
#
# fig = make_subplots(rows=2, cols=3)
#
#
#
# fig.add_trace(
#     go.Line(y=to_display[results_to_display[0]].iloc[0]),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Line(y=to_display[results_to_display[1]].iloc[0]),
#     row=1, col=2
# )
#
# fig.add_trace(
#     go.Line(y=to_display[results_to_display[2]].iloc[0]),
#     row=1, col=2
# )
#
# fig.add_trace(
#     go.Heatmap(z=to_display[results_to_display[3]].iloc[0][-1,:,:],
#                colorbar=dict(x=0.3, len=0.4, y=0.2)),
#     row=2, col=1
# )
#
# fig.add_trace(
#     go.Heatmap(z=to_display[results_to_display[4]].iloc[0][-1,:,:],
#                colorbar=dict(x=0.65, len=0.4, y=0.2)),
#     row=2, col=2
# )
# max_modes = len(to_display[results_to_display[5]].iloc[0][-1,:])
# a_est_x = list(range(max_modes))
# fig.add_trace(
#     go.Bar(x=a_est_x, y=to_display[results_to_display[5]].iloc[0][-1,:]),
#     row=2, col=3
# )
#
# fig.update_layout(title_text="Multiple Subplots with Titles",
#                   showlegend=False
#                  )
# # # Add dropdown
# # plot.update_layout(
# #     updatemenus=[
# #         dict(
# #             buttons=list([
# #                 dict(
# #                     args=["type", "scatter"],
# #                     label="Scatter Plot",
# #                     method="restyle"
# #                 ),
# #                 dict(
# #                     args=["type", "bar"],
# #                     label="Bar Chart",
# #                     method="restyle"
# #                 )
# #             ]),
# #             direction="down",
# #         ),
# #     ]
# # )
#
# fig.show()
