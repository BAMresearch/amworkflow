import logging
import typing
from pathlib import Path

import dolfinx as df
import numpy as np
import pandas as pd
from fenicsxconcrete.finite_element_problem import ConcreteAM, ConcreteThixElasticModel
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import QuadratureEvaluator, ureg

from amworkflow.simulation.experiment import ExperimentProcess, ExperimentStructure

typing.override = lambda x: x


class Simulation:
    """Base class with API for any simulation."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    @typing.override
    def run(self, mesh_file: Path, out_xdmf: Path) -> None:
        """Run simulation and save results in simulation result file (xdmf file).

        Args:
            mesh_file: File path to input mesh file.
            out_xdmf: File path of output xdmf file.

        Returns:

        """
        raise NotImplementedError


class SimulationFenicsXConcrete(Simulation):

    def __init__(
        self,
        pint_parameters: dict,
        **kwargs,
    ) -> None:
        """Simulation class using FenicsXConcrete.
        Args:
            pint_parameters: Dictionary with pint parameters according to FencisXConcrete interface.
        """
        self.pint_parameters = pint_parameters
        super().__init__(**kwargs)

    def parameter_description(self) -> dict[str, str]:
        """description of the required parameters for the simulation"""

        parameter_description = {
            "experiment_type": "type of experiment, possible cases: structure or process",
            "material_type": "type of material, e.g. linear or thixo",

            "for structure simulation": {"top_displacement": "displacement of top surface",},
            "for process simulation": {
            "num_layers": "number of layers",
            "layer_height": "height of each layer",
            "time_per_layer": "time to build one layer (either this or print_velocity)",
            "print_velocity": "print velocity (either this or time_per_layer)",
            "num_time_steps_per_layer": "number of time steps per layer",}
        }

        if self.pint_parameters["experiment_type"].magnitude == "structure":
            parameter_description['ExperimentStructure Parameters'] = ExperimentStructure.parameter_description()
        elif self.pint_parameters["experiment_type"].magnitude == "process":
            parameter_description['ExperimentProcess Parameters'] = ExperimentProcess.parameter_description()

        # if self.pint_parameters["material_type"].magnitude == "linear":
            # parameter_description['Material Parameters'] = LinearElasticity.parameter_description() # not in current FenicsXConcrete version
        if self.pint_parameters["material_type"].magnitude == "thixo":
            parameter_description['Material Parameters'] = ConcreteAM.parameter_description()

        return parameter_description



    def run(self, mesh_file: Path, out_xdmf: Path) -> None:
        """Run simulation and save results in simulation result file (xdmf file).

        Args:
            mesh_file: File path to input mesh file.
            out_xdmf: File path of output xdmf file.

        Returns:

        """

        # add mesh file info to parameter dict
        self.pint_parameters["mesh_file"] = f"{mesh_file}" * ureg("")

        if self.pint_parameters["experiment_type"].magnitude == "structure":

            experiment = ExperimentStructure(self.pint_parameters)
            print('Volume of design', experiment.compute_volume())

            if self.pint_parameters["material_type"].magnitude == "linear":
                self.problem = LinearElasticity(
                    experiment,
                    self.pint_parameters,
                    pv_name=out_xdmf.stem,
                    pv_path=out_xdmf.parent,
                )
            else:
                raise NotImplementedError(
                    "material type not yet implemented: ", self.material_type
                )

            # set sensors
            self.problem.add_sensor(ReactionForceSensor())
            if experiment.p["bc_setting"] == "compr_disp_x" or experiment.p["bc_setting"] == "compr_disp_y":
                self.problem.add_sensor(StressSensor(where=experiment.sensor_location_middle_endge, name="stress sensor"))
                self.problem.add_sensor(StrainSensor(where=experiment.sensor_location_middle_endge, name="strain sensor"))
                self.problem.add_sensor(DisplacementSensor(where=experiment.sensor_location_corner_top, name="disp top"))

            for i in range(0,self.pint_parameters["number_steps"].magnitude):
                d_disp = 0 + (i+1) * self.pint_parameters["top_displacement"] / self.pint_parameters["number_steps"].magnitude
                print('d_disp', d_disp)
                self.problem.experiment.apply_displ_load(d_disp)
                self.problem.solve()
                self.problem.pv_plot()
                print(
                    f"computed disp step: {i}, u_max={self.problem.fields.displacement.x.array[:].max()}, u_min={self.problem.fields.displacement.x.array[:].min()}"
                )
            print(
                "force_x_y_z",
                np.array(self.problem.sensors["ReactionForceSensor"].data),self.problem.sensors["ReactionForceSensor"].units
            )

            ## temporary solution to get equivalent modulus TO BE DELTED
            if self.pint_parameters["bc_setting"].magnitude == "compr_disp_y":
                force_y = np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 1].min()
                u_max = d_disp
                print('equivalent modulus', force_y/(u_max*0.15))
            elif self.pint_parameters["bc_setting"].magnitude == "compr_disp_x":
                force_y = np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 0].min()
                u_max = d_disp
                print('equivalent modulus', force_y/(u_max*0.15))

            # save sensor output as csv file for postprocessing
            if experiment.p["bc_setting"] == "compr_disp_x" or experiment.p["bc_setting"] == "compr_disp_y":
                sensor_dict = {}
                sensor_dict["disp_x"] = np.array(self.problem.sensors["disp top"].data)[:, 0]
                sensor_dict["disp_y"] = np.array(self.problem.sensors["disp top"].data)[:, 1]
                sensor_dict["disp_z"] = np.array(self.problem.sensors["disp top"].data)[:, 2]
                sensor_dict["force_x"] = np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 0]
                sensor_dict["force_y"] = np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 1]
                sensor_dict["force_z"] = np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 2]
                df_sensor = pd.DataFrame(sensor_dict)
                print(df_sensor)
                df_sensor.to_csv(out_xdmf.parent / f"{out_xdmf.stem}.csv", index=False)


        elif self.pint_parameters["experiment_type"].magnitude == "process":
            experiment = ExperimentProcess(self.pint_parameters)

            # define time to build one layer from printer velocity
            # alternative give layer time directly ??
            if "time_per_layer" in self.pint_parameters:
                time_layer = self.pint_parameters["time_per_layer"].magnitude
            else:
                layer_length = (
                    experiment.get_bottom_area() / experiment.p["layer_thickness"]
                )
                time_layer = (
                    layer_length / experiment.p["print_velocity"]
                )  # time to build one layer
                print("layer length approx by surface", layer_length)
                print("time for one layer", time_layer)

            # defining dt and time loading
            self.pint_parameters["dt"] = (
                time_layer / self.pint_parameters["num_time_steps_per_layer"]
            )
            self.pint_parameters["load_time"] = (
                1 * self.pint_parameters["dt"]
            )  # interval where load is applied linear over time

            if self.pint_parameters["material_type"].magnitude == "thixo":
                self.problem = ConcreteAM(
                    experiment,
                    self.pint_parameters,
                    nonlinear_problem=ConcreteThixElasticModel,
                    pv_name=out_xdmf.stem,
                    pv_path=out_xdmf.parent,
                )
            else:
                raise NotImplementedError(
                    "material type not yet implemented: ", self.material_type
                )

            # initial path function describing layer activation
            self.initial_path(
                time_layer,
                t_0=-(self.pint_parameters["num_layers"].magnitude - 1) * time_layer,
            )

            # set sensors
            self.problem.add_sensor(ReactionForceSensor())

            total_time = self.pint_parameters["num_layers"] * time_layer
            while self.problem.time <= total_time.to_base_units().magnitude:
                self.problem.solve()
                self.problem.pv_plot()
                print(
                    f"computed disp t={self.problem.time}, u_max={self.problem.fields.displacement.x.array[:].max()}, u_min={self.problem.fields.displacement.x.array[:].min()}"
                )

            print(
                "force_x",
                np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 0],
            )
            print(
                "force_y",
                np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 1],
            )
            print(
                "force_z",
                np.array(self.problem.sensors["ReactionForceSensor"].data)[:, 2],
            )

        else:
            raise NotImplementedError(
                "requested experiment type not yet implemented: ",
                self.pint_parameters["experiment_type"],
            )

    def initial_path(self, time_layer, t_0: float = 0):
        """Define and set initial path in problem describing layer activation.

        only over height - one layer by time -> negative time, when elements will be reached

        Args:
            time_layer: Time to build one layer.
            t_0: Initial time = -end_time last layer

        Returns:
            path: Path function.

        """

        # get parameters from problem class experiment class would also be possible
        num_layers = self.problem.p["num_layers"]
        layer_height = self.problem.p["layer_height"]

        # init path time array
        q_path = self.problem.rule.create_quadrature_array(self.problem.mesh, shape=1)

        # get quadrature coordinates with work around since tabulate_dof_coordinates()[:] not possible for quadrature spaces!
        V = df.fem.VectorFunctionSpace(
            self.problem.mesh, ("CG", self.problem.p["degree"])
        )
        v_cg = df.fem.Function(V)
        v_cg.interpolate(lambda x: (x[0], x[1], x[2]))
        positions = QuadratureEvaluator(v_cg, self.problem.mesh, self.problem.rule)
        x = positions.evaluate()
        dof_map = np.reshape(x.flatten(), [len(q_path), 3])
        print(np.array(dof_map))
        # save dof_map
        np.savetxt("dof_map.csv", dof_map, delimiter=",")

        # select layers only by layer height - z coordinate
        h_CO = np.array(dof_map)[:, 2]
        h_min = np.arange(0, num_layers * layer_height, layer_height)
        h_max = np.arange(
            layer_height,
            (num_layers + 1) * layer_height,
            layer_height,
        )
        print("h_CO", h_CO)
        print("h_min", h_min)
        print("h_max", h_max)
        new_path = np.zeros_like(q_path)
        EPS = 1e-8
        for i in range(0, len(h_min)):
            layer_index = np.where((h_CO > h_min[i] - EPS) & (h_CO <= h_max[i] + EPS))
            new_path[layer_index] = t_0 + (num_layers - 1 - i) * time_layer

        q_path = new_path

        # set new path
        self.problem.set_initial_path(q_path)

        return