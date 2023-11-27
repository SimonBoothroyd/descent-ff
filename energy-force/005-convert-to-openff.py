import pathlib
import click
import numpy
import openff.toolkit
import smee
import torch
from openff.toolkit.typing.engines.smirnoff import ParameterHandler


@click.command()
@click.argument(
    "force_field_path",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
def main(force_field_path: pathlib.Path):
    force_field_openff = openff.toolkit.ForceField("openff-2.1.0.offxml")
    force_field_smee: smee.TensorForceField = torch.load(force_field_path)

    potential_types = ["Bonds", "Angles", "ProperTorsions"]
    potential_attrs = {
        "Bonds": ["length", "k"],
        "Angles": ["angle", "k"],
        "ProperTorsions": ["periodicity", "phase", "k"],
    }

    for potential_type in potential_types:
        potential = force_field_smee.potentials_by_type[potential_type]
        handler: ParameterHandler = force_field_openff.get_parameter_handler(
            potential_type
        )

        for key, parameter in zip(potential.parameter_keys, potential.parameters):
            smirks = key.id

            handler_parameters = handler.get_parameter({"smirks": smirks})
            assert len(handler_parameters) == 1
            handler_parameter = handler_parameters[0]

            changes = ""

            for attr in potential_attrs[potential_type]:
                attr_idx = potential.parameter_cols.index(attr)
                attr_value = parameter[attr_idx].item()

                handler_attr = attr

                if key.mult is not None:
                    handler_attr = f"{handler_attr}{key.mult + 1}"

                if attr not in {"periodicity", "phase", "idivf"}:
                    orig = getattr(handler_parameter, handler_attr).m_as(
                        potential.parameter_units[attr_idx]
                    )

                    if orig != 0.0:
                        pct_change = (attr_value - orig) / orig * 100
                    else:
                        pct_change = numpy.nan

                    changes += (
                        f"{attr} "
                        f"before={orig:.2f} "
                        f"after={attr_value:.2f} "
                        f"delta={pct_change:.2f}%  |  "
                    )

                setattr(
                    handler_parameter,
                    handler_attr,
                    attr_value * potential.parameter_units[attr_idx],
                )

            print(handler_parameter.id, changes)

    force_field_openff.to_file("openff-2.1.0-smee.offxml")


if __name__ == "__main__":
    main()
