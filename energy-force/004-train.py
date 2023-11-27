"""Train the force field."""
import contextlib
import datetime
import math
import os
import pathlib

import datasets
import datasets.distributed
import datasets.table
import pydantic
import smee
import tensorboardX
import torch
import torch.distributed

import descent.optim
import descent.targets.energy
import descent.utils.loss
import descent.utils.reporting

MISSING_SMILES = set()

WORLD_SIZE = torch.multiprocessing.cpu_count() - 2


@contextlib.contextmanager
def open_writer(path: pathlib.Path, rank: int) -> tensorboardX.SummaryWriter:
    if rank != 0:
        yield None
    else:
        path.mkdir(parents=True, exist_ok=True)

        with tensorboardX.SummaryWriter(str(path)) as writer:
            yield writer


class ParameterConfig(pydantic.BaseModel):
    """Configuration for how a potential's parameters should be trained."""

    cols: list[str] = pydantic.Field(
        description="The parameters to train, e.g. 'k', 'length', 'epsilon'."
    )

    scales: dict[str, float] = pydantic.Field(
        {},
        description="The scales to apply to each parameter, e.g. 'k': 1.0, "
        "'length': 1.0, 'epsilon': 1.0.",
    )
    constraints: dict[str, tuple[float | None, float | None]] = pydantic.Field(
        {},
        description="The min and max values to clamp each parameter within, e.g. "
        "'k': (0.0, None), 'angle': (0.0, pi), 'epsilon': (0.0, None), where "
        "none indicates no constraint.",
    )


class TrainableParameters:
    """A wrapper around a SMEE force field that handles zeroing out gradients of
    fixed parameters and applying parameter constraints."""

    def __init__(
        self,
        force_field: smee.TensorForceField,
        parameters: dict[str, ParameterConfig],
    ):
        self.potential_types = [*parameters]
        self._force_field = force_field

        potentials = [
            force_field.potentials_by_type[potential_type]
            for potential_type in self.potential_types
        ]

        self._frozen_cols = [
            [
                i
                for i, col in enumerate(potential.parameter_cols)
                if col not in parameters[potential_type].cols
            ]
            for potential_type, potential in zip(self.potential_types, potentials)
        ]

        self._scales = [
            torch.tensor(
                [
                    parameters[potential_type].scales.get(col, 1.0)
                    for col in potential.parameter_cols
                ]
            ).reshape(1, -1)
            for potential_type, potential in zip(self.potential_types, potentials)
        ]
        self._constraints = [
            {
                i: parameters[potential_type].constraints[col]
                for i, col in enumerate(potential.parameter_cols)
                if col in parameters[potential_type].constraints
            }
            for potential_type, potential in zip(self.potential_types, potentials)
        ]

        self.parameters = [
            (potential.parameters.detach().clone() * scale).requires_grad_()
            for potential, scale in zip(potentials, self._scales)
        ]

    @property
    def force_field(self) -> smee.TensorForceField:
        for potential_type, parameter, scale in zip(
            self.potential_types, self.parameters, self._scales
        ):
            potential = self._force_field.potentials_by_type[potential_type]
            potential.parameters = parameter / scale

        return self._force_field

    @torch.no_grad()
    def clamp(self):
        for parameter, constraints in zip(self.parameters, self._constraints):
            for i, (min_value, max_value) in constraints.items():
                if min_value is not None:
                    parameter[:, i].clamp_(min=min_value)
                if max_value is not None:
                    parameter[:, i].clamp_(max=max_value)

    @torch.no_grad()
    def freeze_grad(self):
        for parameter, col_idxs in zip(self.parameters, self._frozen_cols):
            parameter.grad[:, col_idxs] = 0.0


def write_metrics(
    i: int,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    prior_k_torsion: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
):
    print(f"epoch={i} loss={loss:.6f}", flush=True)

    writer.add_scalar("loss", loss.detach().item(), i)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), i)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), i)

    writer.add_scalar("prior_k_torsion", prior_k_torsion.detach().item(), i)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), i)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), i)
    writer.flush()


def main(rank: int):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)

    sources = ["gen2-opt", "gen2-torsion", "spice-des-monomers", "spice-pubchem"]
    # sources = ["gen2-torsion"]

    force_field, topologies = torch.load("outputs/openff-2.1.0.pt")

    n_epochs = 1000
    lr = 0.01

    trainable = TrainableParameters(
        force_field,
        {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0 / 100.0, "length": 1.0},
                constraints={"k": (0.0, None), "length": (0.0, None)},
            ),
            "Angles": ParameterConfig(
                cols=["k", "angle"],
                scales={"k": 1.0 / 100.0, "angle": 1.0},
                constraints={"k": (0.0, None), "angle": (0.0, math.pi)},
            ),
            "ProperTorsions": ParameterConfig(cols=["k"], scales={"k": 1.0}),
        },
    )

    dataset = datasets.concatenate_datasets(
        [
            datasets.Dataset.load_from_disk(f"outputs/data-clustered/{source}")
            for source in sources
        ]
    )
    n_entries = len(dataset)

    unique_smiles = descent.targets.energy.extract_smiles(dataset)
    topologies = {k: v for k, v in topologies.items() if k in unique_smiles}

    dataset = datasets.distributed.split_dataset_by_node(
        dataset, rank=rank, world_size=WORLD_SIZE
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = pathlib.Path(f"outputs/runs/{timestamp}")

    with open_writer(experiment_dir, rank) as writer:
        optimizer = torch.optim.Adam(trainable.parameters, lr=lr, amsgrad=True)

        if rank == 0:
            for v in tensorboardX.writer.hparams({"optimizer": "Adam", "lr": lr}, {}):
                writer.file_writer.add_summary(v)

        for i in range(n_epochs):
            e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                dataset, trainable.force_field, topologies, "mean"
            )

            loss_energy = ((e_pred - e_ref) ** 2).sum() / n_entries
            loss_forces = ((f_pred - f_ref) ** 2).sum() / n_entries

            # k_col_torsion = trainable.force_field.potentials_by_type[
            #     "ProperTorsions"
            # ].parameter_cols.index("k")
            # prior_k_torsion = (
            #     trainable.force_field.potentials_by_type["ProperTorsions"]
            #     .parameters[:, k_col_torsion]
            #     .square()
            #     .mean()
            # )
            prior_k_torsion = torch.tensor(0.0)

            loss = loss_energy + loss_forces + prior_k_torsion
            loss.backward()

            torch.distributed.all_reduce(loss)
            torch.distributed.all_reduce(loss_energy)
            torch.distributed.all_reduce(loss_forces)
            # torch.distributed.all_reduce(prior_k_torsion)

            for parameter in trainable.parameters:
                torch.distributed.all_reduce(parameter.grad)

            trainable.freeze_grad()

            if rank == 0:
                write_metrics(
                    i, loss, loss_energy, loss_forces, prior_k_torsion, writer
                )

            optimizer.step()
            optimizer.zero_grad()

            trainable.clamp()

            if rank == 0 and i % 100 == 0:
                torch.save(
                    trainable.force_field, experiment_dir / f"force-field-epoch-{i}.pt"
                )

    if rank != 0:
        exit(0)

    for potential_type in trainable.potential_types:
        descent.utils.reporting.print_potential_summary(
            force_field.potentials_by_type[potential_type]
        )
        print("")

    torch.save(force_field, experiment_dir / "force-field.pt")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.multiprocessing.spawn(main, nprocs=WORLD_SIZE, join=True)
