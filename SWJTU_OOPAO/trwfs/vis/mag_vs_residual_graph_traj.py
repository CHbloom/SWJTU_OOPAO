import pickle
from pathlib import Path

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import pandas as pd
from pypet import Trajectory

class WFS_SIMU_VISU():

    def __init__(self, filename):
        self.traj = Trajectory("run_loops")
        self.traj.f_load(load_parameters=2, load_derived_parameters=0, load_results=1,
                        load_other_data=0, filename=filename, force=True)
        self.traj.v_auto_load = True

        self.cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.cycle_styles = ['-', '--', '-.']


    def get_explored_parameters_and_combinations(self):
        ep = self.traj.f_get_explored_parameters()
        eps_that_change = []
        for p in ep.keys():
            if len(np.unique(ep[p].f_get_range())) > 1:
                eps_that_change.append(p)

        eps_dict = {el:[] for el in eps_that_change}
        for p in eps_that_change:
            eps_dict[p] = ep[p].f_get_range()

        df = pd.DataFrame.from_dict(eps_dict)
        df.drop_duplicates()

        return df

    def get_graph(self, y, parameters, remove_first=50):
        parameters_in_file = self.traj.f_get_parameters()
        all_fields = y + parameters
        # The data we will use to create the plot will be stored in "data" dict
        data = {el: [] for el in all_fields}
        # Run through all runs
        for run_name in self.traj.f_get_run_names():
            self.traj.f_set_crun(run_name)
            # Grab only the fields we are interested in
            for f in all_fields:
                if f in parameters_in_file.keys():
                    if isinstance(self.traj[f], list):
                        data[f].append((self.traj[f][remove_first:]))
                    else:
                        data[f].append(self.traj[f])
                else:  # When the field is a result, need to access crun
                    if isinstance(self.traj.crun[f], np.ndarray):
                         data[f].append((self.traj.crun[f][remove_first:]))
                    else:
                        data[f].append(self.traj.crun[f])

        df = pd.DataFrame.from_dict(data)

        return df

    def fetch_plot_data(self, x, y, selected_parameter_values, remove_first=50, reduce_lines=[], reduce_func=np.max):

        filter_function = lambda *args: all([arg == value for (arg, value) in zip(args, list(selected_parameter_values.values()))])


        idx_iterator = self.traj.f_find_idx(list(selected_parameter_values.keys()), filter_function)

        all_fields = [x] + [y] + list(selected_parameter_values.keys())
        data = {el: [] for el in all_fields}

        for idx in idx_iterator:
            self.traj.v_idx = idx
            for f in all_fields:
                if f.startswith("parameters."):
                    data[f].append(self.traj[f])
                else:  # When the field is a result, need to access crun
                    data[f].append(self.traj.crun[f])

        df = pd.DataFrame.from_dict(data)

        try:
            # columns are the different lines
            parameters_to_use = list(selected_parameter_values.keys()).copy()
            to_plot = df.pivot(index=x, columns=parameters_to_use, values=y)

            if len(reduce_lines) != 0:
                parameters_to_use = [elem for elem in parameters_to_use if elem not in reduce_lines]
                to_plot = df.pivot_table(index=x, columns=parameters_to_use, values=y, aggfunc=reduce_func)

        except ValueError as e:
            print(
                "ERROR: There seems to be more parameters that expected, cannot create a plot because there are duplicate values")
            # print(f"The used parameters during the run were: {parameters_in_file.keys()}")
            print("Did you forget to use one?")
            print(e)
            return None

        return df, to_plot


    def display_graph_parameters(self, x, y, parameters, remove_first=50, reduce_function=np.mean, reduce_lines=[], aggfunc=np.max, logx=False, logy=False, title=None, xlabel=None, ylabel=None, config_subtitle=[]):
        parameters_in_file = self.traj.f_get_parameters()
        all_fields = [x] + [y] + parameters
        # The data we will use to create the plot will be stored in "data" dict
        data = {el:[] for el in all_fields}
        # Run through all runs
        for run_name in self.traj.f_get_run_names():
            self.traj.f_set_crun(run_name)
            # Grab only the fields we are interested in
            for f in all_fields:
                if "parameters."+f in parameters_in_file.keys():
                    if isinstance(self.traj[f], list):
                        data[f].append(reduce_function(self.traj[f][remove_first:]))
                    else:
                        data[f].append(self.traj[f])
                else: # When the field is a result, need to access crun
                    if isinstance(self.traj.crun[f], np.ndarray):
                        data[f].append(reduce_function(self.traj.crun[f][remove_first:]))
                    else:
                        data[f].append(self.traj.crun[f])
                        
        df = pd.DataFrame.from_dict(data)



        try :
            # columns are the different lines
            to_plot = df.pivot(index=x, columns=parameters, values=y)
            parameters_to_use = parameters.copy()
            if len(reduce_lines) != 0:
                parameters_to_use = [ elem for elem in parameters if elem not in reduce_lines]
                to_plot = df.pivot_table(index=x , columns=parameters_to_use, values=y, aggfunc=aggfunc )


        except ValueError as e:
            print("ERROR: There seems to be more parameters that expected, cannot create a plot because there are duplicate values")
            #print(f"The used parameters during the run were: {parameters_in_file.keys()}")
            print("Did you forget to use one?")
            print(e)
            exit(1)

       
        gin = df.groupby('gainCL').size().to_frame().rename(columns={0:'gainCL'})


        print(to_plot)



        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #styles = self.cycle_styles[:df[parameters[1], parameters[2]].nunique()]
        styles = self.cycle_styles[:(~df.duplicated(parameters_to_use[1:])).sum()]
        #styles = self.cycle_styles

        colors = self.cycle_colors[:df[parameters_to_use[0]].nunique()]
        cc = (cycler(color=colors) *
              cycler(linestyle=styles))

        fig, ax = plt.subplots()
        ax.set_prop_cycle(cc)
        if not xlabel:
            xlabel = x
        if not ylabel:
            ylabel = y

        to_plot.plot(kind='line', grid=True, xlabel=xlabel, ylabel=ylabel, ax=ax, logx=logx, logy=logy)#, marker='o')

        if title:
            subtitle = ""
            for conf_idx in range(len(config_subtitle)):
                conf = config_subtitle[conf_idx]
                if conf_idx >= (len(config_subtitle)-1):
                    subtitle += f" {conf}={self.traj[conf]}"
                else:
                    subtitle += f" {conf}={self.traj[conf]},"
            ax.set_title(title+"\n"+subtitle)

        #for k, v in to_plot.iterrows():
        #    ax.annotate(k, v,  xytext=(10,-5), textcoords='offset points', family='sans-serif', fontsize=18, color='darkslategrey')
        
		
        lines = ax.get_lines()
        legend_title = f"({parameters_to_use} )"
        leg = ax.legend(title=legend_title)
        lined = {}  # Will map legend lines to original lines.
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(True)  # Enable picking on the legend line.
            lined[legline] = origline

        def on_pick(event):
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled.
            legline.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)


        self.traj.f_restore_default()

    def display_mag_residual_gainCL(self, reduce_lines=None):
        data = {"mag": [], "gainCL": [],"Avg SR": [], "Avg Residual (nm)": [], "Avg Turbulence (nm)": [], "enable_custom_frames": []}
        removed_first = 50
        loops = 1000
        for run_name in self.traj.f_get_run_names():
            self.traj.f_set_crun(run_name)
            data["Avg SR"].append(np.mean(self.traj.crun.SR_H[removed_first:]))
            data["Avg Residual (nm)"].append(np.mean(self.traj.crun.Residual[removed_first:]))
            data["Avg Turbulence (nm)"].append(np.mean(self.traj.crun.Turbulence[removed_first:]))
            data["mag"].append(self.traj.magnitude)
            data["gainCL"].append(self.traj.gainCL)
            data["enable_custom_frames"].append(self.traj.enable_custom_frames)
            loops = len(self.traj.crun.SR_H)

        df = pd.DataFrame.from_dict(data)
        to_plot_custom = (df.loc[df['enable_custom_frames'] == True]).pivot(index='mag', columns='gainCL', values='Avg SR')
        to_plot_normal = (df.loc[df['enable_custom_frames'] == False]).pivot(index='mag', columns='gainCL', values='Avg SR')

        if reduce_lines == "min":
            to_keep = list(np.unique(to_plot_custom.idxmin(axis="columns")))
            to_plot_custom = to_plot_custom[to_keep]

            to_keep = list(np.unique(to_plot_normal.idxmin(axis="columns")))
            to_plot_normal = to_plot_normal[to_keep]
        elif reduce_lines == "max":
            to_keep = list(np.unique(to_plot_custom.idxmax(axis="columns")))
            to_plot_custom = to_plot_custom[to_keep]

            to_keep = list(np.unique(to_plot_normal.idxmax(axis="columns")))
            to_plot_normal = to_plot_normal[to_keep]

        ax = to_plot_normal.plot(kind='line', linestyle='-', label="Normal")
        ax = to_plot_custom.plot(kind='line', ax=ax, linestyle='--', label="TR48")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Strehl Ratio (H band)")
        ax.set_title(f"Average Strehl ratio as a funtion of the magnitude.\nr0=0.186, {loops} closed loop steps, first {removed_first} removed")

        plt.show()

        self.traj.f_restore_default()


    def display_mag_residual_gainCL2(self, reduce_lines=None):
        data = {"mag": [], "gainCL": [],"Avg SR": [], "Avg Residual (nm)": [], "Avg Turbulence (nm)": [], "enable_custom_frames": [], "modulation": []}
        removed_first = 50
        loops = 1000
        for run_name in self.traj.f_get_run_names():
            self.traj.f_set_crun(run_name)
            data["Avg SR"].append(np.mean(self.traj.crun.SR_H[removed_first:]))
            data["Avg Residual (nm)"].append(np.mean(self.traj.crun.Residual[removed_first:]))
            data["Avg Turbulence (nm)"].append(np.mean(self.traj.crun.Turbulence[removed_first:]))
            data["mag"].append(np.mean(self.traj.magnitude))
            data["gainCL"].append(np.mean(self.traj.gainCL))
            data["enable_custom_frames"].append(np.mean(self.traj.enable_custom_frames))
            data["modulation"].append(np.mean(self.traj.modulation))
            loops = len(self.traj.crun.SR_H)

        df = pd.DataFrame.from_dict(data)
        to_plot_custom = (df.loc[df['modulation'] == 3]).pivot(index='mag', columns='gainCL', values='Avg SR')
        to_plot_normal = (df.loc[df['modulation'] == 0]).pivot(index='mag', columns='gainCL', values='Avg SR')

        if reduce_lines == "min":
            to_keep = list(np.unique(to_plot_custom.idxmin(axis="columns")))
            to_plot_custom = to_plot_custom[to_keep]

            to_keep = list(np.unique(to_plot_normal.idxmin(axis="columns")))
            to_plot_normal = to_plot_normal[to_keep]
        elif reduce_lines == "max":
            to_keep = list(np.unique(to_plot_custom.idxmax(axis="columns")))
            to_plot_custom = to_plot_custom[to_keep]

            to_keep = list(np.unique(to_plot_normal.idxmax(axis="columns")))
            to_plot_normal = to_plot_normal[to_keep]

        ax = to_plot_normal.plot(kind='line', linestyle='-', label="Normal")
        ax = to_plot_custom.plot(kind='line', ax=ax, linestyle='--', label="TR48")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Strehl Ratio (H band)")
        ax.set_title(f"Average Strehl ratio as a funtion of the magnitude.\nr0=0.186, {loops} closed loop steps, first {removed_first} removed")

        plt.show()

        self.traj.f_restore_default()

    def show_specific_closed_loop(self, mag, gain):

        data_cl = {"Turbulence": [], "Residual Normal": [],"Residual Custom": []}

        filter_function  = lambda magnitude, gainCL: magnitude==mag and gainCL==gain
        idx_iterator = self.traj.f_find_idx(['parameters.magnitude', 'parameters.gainCL'], filter_function)
        for idx in idx_iterator:
            self.traj.v_idx = idx
            data_cl["Turbulence"] = self.traj.crun.Turbulence
            if self.traj.enable_custom_frames:
                data_cl["Residual Custom"] = self.traj.crun.Residual
            else:
                data_cl["Residual Normal"] = self.traj.crun.Residual

        df_cl = pd.DataFrame.from_dict(data_cl)
        ax = df_cl.plot()
        ax.set_xlabel("Time (iterations)")
        ax.set_ylabel("WFE (nm)")
        ax.set_title(f"Closed-loop WFE through iterations.\nMagnitude={mag}, Gain={gain}")

        plt.show()
        self.traj.f_restore_default()


    def show_specific_closed_loop2(self, mag, gain):

        data_cl = {"Turbulence": [], "Residual Normal": [],"Residual Custom": []}

        filter_function  = lambda magnitude, gainCL: magnitude==mag and gainCL==gain
        idx_iterator = self.traj.f_find_idx(['parameters.magnitude', 'parameters.gainCL'], filter_function)
        for idx in idx_iterator:
            self.traj.v_idx = idx
            data_cl["Turbulence"] = self.traj.crun.Turbulence
            if self.traj.modulation == 5:
                data_cl["Residual Custom"] = self.traj.crun.Residual
            elif self.traj.modulation == 0:
                data_cl["Residual Normal"] = self.traj.crun.Residual

        df_cl = pd.DataFrame.from_dict(data_cl)
        ax = df_cl.plot()
        ax.set_xlabel("Time (iterations)")
        ax.set_ylabel("WFE (nm)")
        ax.set_title(f"Closed-loop WFE through iterations.\nMagnitude={mag}, Gain={gain}")

        plt.show()
        self.traj.f_restore_default()

    def show_specific_closed_loop3(self, mag, gain):

        data_cl = {"Turbulence": [], "Residual Normal": [],"Residual Custom": []}

        filter_function  = lambda magnitude, gainCL: magnitude==mag and gainCL==gain
        idx_iterator = self.traj.f_find_idx(['parameters.magnitude', 'parameters.gainCL'], filter_function)
        for idx in idx_iterator:
            self.traj.v_idx = idx
            data_cl["Turbulence"] = self.traj.crun.Turbulence
            if self.traj.modulation == 5 and self.traj.enable_custom_frames:
                data_cl["Residual Custom"] = self.traj.crun.Residual
                print(self.traj.crun.numRemFramesPerBase)
            if self.traj.modulation == 5 and (not self.traj.enable_custom_frames):
                data_cl["Residual Modulated"] = self.traj.crun.Residual
            elif self.traj.modulation == 0:
                data_cl["Residual Normal"] = self.traj.crun.Residual

        df_cl = pd.DataFrame.from_dict(data_cl)
        ax = df_cl.plot()
        ax.set_xlabel("Time (iterations)")
        ax.set_ylabel("WFE (nm)")
        ax.set_title(f"Closed-loop WFE through iterations.\nMagnitude={mag}, Gain={gain}")


        self.traj.f_restore_default()


    def show_specific_closed_loop4(self, mag, gain):

        data_cl = {"Turbulence": [], "Residual Modulated": [],"Residual Custom": []}

        filter_function  = lambda magnitude, gainCL: magnitude==mag and gainCL==gain
        idx_iterator = self.traj.f_find_idx(['parameters.magnitude', 'parameters.gainCL'], filter_function)
        for idx in idx_iterator:
            self.traj.v_idx = idx
            data_cl["Turbulence"] = self.traj.crun.Turbulence
            if self.traj.modulation == 5 and self.traj.enable_custom_frames:
                data_cl["Residual Custom"] = self.traj.crun.Residual
            if self.traj.modulation == 5 and (not self.traj.enable_custom_frames):
                data_cl["Residual Modulated"] = self.traj.crun.Residual

        df_cl = pd.DataFrame.from_dict(data_cl)
        ax = df_cl.plot()
        ax.set_xlabel("Time (iterations)")
        ax.set_ylabel("WFE (nm)")
        ax.set_title(f"Closed-loop WFE through iterations.\nMagnitude={mag}, Gain={gain}")


        self.traj.f_restore_default()


if __name__ == "__main__":

    #WSV = WFS_SIMU_VISU("/media/simonc/Sagittarius/Programming/OOPAO_TRWFS/trwfs/sum/res/general_right_before_leaving_small_shortcut.hdf5")
    WSV = WFS_SIMU_VISU("/mnt/home/usager/cars2019/Documents/Programming/OOPAO_TRWFS/trwfs/sum/res/r0_sweeping_09mar2024.hdf5")
    WSV.display_graph_parameters(x="photons_per_subArea",
                                 y="SR_H", #SR_H
                                 remove_first=1000,
                                 parameters=[ "enable_custom_frames","modulation", "gainCL", "r0"],
                                 reduce_lines=["gainCL"],
                                 aggfunc=np.max,
                                 logx=True,
                                 ylabel="Strehl Ratio (H band)",
                                 xlabel="Photons per subaperture",
                                 title="Strehl Ratio as a function of photons per subaperture",
                                 config_subtitle=["diameter", "r0", "lightThreshold"])
    #WSV.display_mag_residual_gainCL2()
    #
    #WSV.show_specific_closed_loop(mag=15.0, gain=0.1)
    # WSV.show_specific_closed_loop3(mag=16.5, gain=0.3)
    # WSV.show_specific_closed_loop3(mag=17.0, gain=0.3)
    #WSV.show_specific_closed_loop4(mag=12.0, gain=0.4)


    plt.show()

# plt.figure()
#
# for cl in [0.1,0.2,0.3]:
#     tmp = df.loc[df['gainCL'] == cl]
#     tmp.plot(x="mag", y="Avg SR", label=f"LG={cl} Custom",)

# for run_name in self.traj.f_get_run_names():
#     self.traj.f_set_crun(run_name)
#     print(self.traj.SR)

# filter_function  = lambda enable_custom_frames: enable_custom_frames==True
# idx_iterator = traj.f_find_idx(['parameters.enable_custom_frames'], filter_function)
# for idx in idx_iterator:
#     # We focus on one particular run. This is equivalent to calling `traj.f_set_crun(idx)`.
#     traj.v_idx = idx
