import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import pandas as pd
from pypet import Trajectory

class WFS_SIMU_VISU():

    def __init__(self, filename):
        self.traj = Trajectory("run_loops")
        self.traj.f_load(load_parameters=2, load_derived_parameters=0, load_results=1,
                        load_other_data=0, filename=filename)
        self.traj.v_auto_load = True

        self.cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.cycle_styles = ['--', '-', ':', '-.']

    def display_graph_parameters(self, x, y, parameters, remove_first=50, reduce_function=np.mean, reduce_lines=False, logx=False, logy=False, title=None, xlabel=None, ylabel=None, config_subtitle=[]):
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

            if reduce_lines:
                to_keep = list(np.unique(to_plot.idxmax(axis="columns")))
                to_plot = to_plot[to_keep]

        except ValueError as e:
            print("ERROR: There seems to be more parameters that expected, cannot create a plot because there are duplicate values")
            #print(f"The used parameters during the run were: {parameters_in_file.keys()}")
            print("Did you forget to use one?")
            print(e)
            exit(1)


        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        styles = self.cycle_styles[:df[parameters[1]].nunique()]
        #styles = self.cycle_styles

        colors = self.cycle_colors[:df[parameters[0]].nunique()]
        cc = (cycler(color=colors) *
              cycler(linestyle=styles))

        fig, ax = plt.subplots()
        ax.set_prop_cycle(cc)
        if not xlabel:
            xlabel = x
        if not ylabel:
            ylabel = y

        to_plot.plot(kind='line', grid=True, xlabel=xlabel, ylabel=ylabel, ax=ax, logx=logx, logy=logy)

        if title:
            subtitle = ""
            for conf_idx in range(len(config_subtitle)):
                conf = config_subtitle[conf_idx]
                if conf_idx >= (len(config_subtitle)-1):
                    subtitle += f" {conf}={self.traj[conf]}"
                else:
                    subtitle += f" {conf}={self.traj[conf]},"
            ax.set_title(title+"\n"+subtitle)

        plt.show()

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

        data_cl = {"Turbulence": [], "Residual with mod": [],"Residual No mod": []}

        filter_function  = lambda magnitude, gainCL: magnitude==mag and gainCL==gain
        idx_iterator = self.traj.f_find_idx(['parameters.magnitude', 'parameters.gainCL'], filter_function)
        for idx in idx_iterator:
            self.traj.v_idx = idx
            data_cl["Turbulence"] = self.traj.crun.Turbulence
            if self.traj.modulation == 3 and self.traj.enable_custom_frames == False:
                data_cl["Residual with mod"] = self.traj.crun.Residual
            elif self.traj.modulation == 0 and self.traj.enable_custom_frames == False:
                data_cl["Residual No mod"] = self.traj.crun.Residual

        df_cl = pd.DataFrame.from_dict(data_cl)
        ax = df_cl.plot()
        ax.set_xlabel("Time (iterations)")
        ax.set_ylabel("WFE (nm)")
        ax.set_title(f"Closed-loop WFE through iterations.\nMagnitude={mag}, Gain={gain}")

        plt.show()
        self.traj.f_restore_default()

if __name__ == "__main__":

    WSV = WFS_SIMU_VISU("/mnt/home/usager/cars2019/Documents/Programming/OOPAO_TRWFS/trwfs/sum/res/long_test_29dec2023.hdf5")
    WSV.display_graph_parameters(x="photons_per_subArea",
                                 y="SR_H",
                                 parameters=["modulation","enable_custom_frames", "gainCL"],
                                 remove_first=100,
                                 reduce_lines=["gainCL"],
                                 aggfunc=np.max,
                                 logx=True,
                                 ylabel="Strehl Ratio (H band)",
                                 xlabel="Photons per subaperture",
                                 title="Strehl Ratio as a function of photons per subaperture",
                                 config_subtitle=["r0", "diameter"])
    #WSV.display_mag_residual_gainCL2()
    #
    #WSV.show_specific_closed_loop2(mag=10.0, gain=0.3)




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
