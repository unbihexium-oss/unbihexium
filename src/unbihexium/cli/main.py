# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/cli/main.py
# Title       : Command line interface
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires click and rich; model
#               commands need PyTorch or ONNX Runtime
# =============================================================================
#
# Abstract
# --------
# The `unbihexium` command:
#
#   info                         library version and registry sizes
#   zoo list|info|build|export|verify|where|clear
#                                browse the model catalogue and manage the
#                                local model store
#   train MODEL --data DIR       train or fine-tune a model (or --synthetic N
#                                to check the setup on generated data)
#   evaluate MODEL --data DIR    accuracy of a model on a dataset split
#   predict MODEL INPUT OUTPUT   run a model on a raster and write GeoJSON,
#                                GeoTIFF or JSON depending on the task
#   pipeline list|run            registered processing pipelines
#   index NAME -i IN -o OUT      spectral index of a raster
#   serve [--host H] [--port P]  REST service (unbihexium.serving) with
#                                uvicorn; needs the serving extra
#
# Model arguments accept a catalogue family or model id (starter weights), a
# checkpoint written by `train` (.pt) or an ONNX export (.onnx).
#
# Exit status
# -----------
#   0  success
#   1  invalid input, missing files or a failed command
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON output of listings and metrics.
import json

# Represent file paths.
from pathlib import Path

# Type of loosely structured values, and the type of functions that never return.
from typing import Any, NoReturn

# Command line framework.
import click

# Rich terminal output.
from rich.console import Console

# Escape text that must not be read as rich markup, such as "[torch]".
from rich.markup import escape

# Tables in the terminal.
from rich.table import Table

# Version of the package.
from unbihexium._version import __version__

# Console used by every command.
console = Console()

# Console for warnings, which go to standard error.
err_console = Console(stderr=True)

# Install hint for commands that need PyTorch.
TORCH_HINT = "install PyTorch with pip install 'unbihexium[torch]'"


# Print an error and exit with status 1.
def fail(message: str) -> NoReturn:
    # Error message in red; the text itself is not markup.
    console.print(f"[red]Error:[/] {escape(message)}")
    # Non-zero exit status.
    raise SystemExit(1)


# Print a warning to standard error.
def warn(message: str) -> None:
    # Warning in yellow; the text itself is not markup.
    err_console.print(f"[yellow]Warning:[/] {escape(message)}")


# Variant for a model argument: the option, else the configured default for
# bare catalogue family names, else None (ids, files and checkpoints).
def resolve_variant(model: str, variant: str | None) -> str | None:
    # An explicit option wins.
    if variant is not None:
        # Use it.
        return variant
    # Catalogue families, imported lazily.
    from unbihexium.zoo import list_specs

    # Only bare family names take the configured default.
    if model not in {spec.family for spec in list_specs()}:
        # Ids with a variant suffix, files and checkpoints.
        return None
    # Settings, imported lazily.
    from unbihexium.config import get_settings

    # Configured default variant; invalid configurations are reported.
    try:
        # Layered settings.
        return get_settings().model.variant
    # Invalid files or environment variables.
    except ValueError as exc:
        # Report and exit.
        fail(str(exc))


# Warn when a catalogue starter model that needs training is run.
def warn_if_untrained(model: Any, variant: str | None = None) -> None:
    # Files and model objects are the user's own trained models.
    if not isinstance(model, str) or Path(model).suffix:
        # Nothing to say.
        return
    # Catalogue lookup, imported lazily.
    from unbihexium.zoo import get_model

    # Entry of the id, or of the family with the variant.
    entry = get_model(f"{model}_{variant}" if variant else model) or get_model(model)
    # Starter models of trainable tasks produce meaningless output.
    if entry is not None and entry.requires_training:
        # One-line notice.
        warn(
            f"{entry.model_id} is an untrained starter model; its output has no meaning "
            "until the model is trained (see unbihexium train)"
        )  # End of the warning.


# Print a dictionary as JSON; NaN becomes null.
def print_json(data: Any) -> None:
    # Convert non-finite floats for strict JSON.
    text = json.dumps(data, indent=2, default=str)
    # Strict JSON has no NaN or infinity.
    text = text.replace("NaN", "null").replace("Infinity", "null")
    # Plain output without rich markup.
    click.echo(text)


# Root command group.
@click.group(help="Unbihexium: Earth observation, geospatial, remote sensing and SAR library.")
@click.version_option(version=__version__, prog_name="unbihexium")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    # Shared state of the subcommands.
    ctx.ensure_object(dict)
    # Verbose flag.
    ctx.obj["verbose"] = verbose
    # Library log handler; the level comes from UNBIHEXIUM_LOG_LEVEL unless
    # --verbose asks for debug messages.
    from unbihexium.utils.log import configure_logging

    # Messages go to standard error.
    try:
        # DEBUG with --verbose, otherwise the environment or WARNING.
        configure_logging("DEBUG" if verbose else None)
    # Unknown level names in the environment.
    except ValueError as exc:
        # Report and exit.
        fail(f"invalid UNBIHEXIUM_LOG_LEVEL: {exc}")


# Library information.
@main.command(help="Display library information.")
def info() -> None:
    # Registries are imported lazily to keep start-up fast.
    import unbihexium.ai
    from unbihexium.registry.capabilities import CapabilityRegistry  # Capabilities.
    from unbihexium.registry.pipelines import PipelineRegistry  # Pipelines.
    from unbihexium.zoo import catalog_version, list_models  # Model zoo.

    # Version line.
    console.print(f"[bold blue]Unbihexium[/] v{__version__}")
    # Registered capabilities.
    console.print(f"Registered capabilities: {len(CapabilityRegistry.ids())}")
    # Model zoo size.
    console.print(f"Model zoo models: {len(list_models())} (catalogue {catalog_version()})")
    # Registered pipelines.
    console.print(f"Registered pipelines: {len(PipelineRegistry.ids())}")


# Model zoo command group.
@main.group(help="Browse the model catalogue and manage the local model store.")
def zoo() -> None:
    # Group without its own behaviour.
    pass


# List models.
@zoo.command("list", help="List model zoo models.")
@click.option("--task", "-t", help="Filter by task, for example detection.")
@click.option("--domain", "-d", help="Filter by capability domain.")
@click.option("--variant", help="Filter by variant: tiny, base, large or mega.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def zoo_list(task: str | None, domain: str | None, variant: str | None, as_json: bool) -> None:
    # Imported lazily.
    from unbihexium.zoo import list_models

    # Matching registry entries.
    try:
        # Filtered listing.
        entries = list_models(task=task, domain=domain, variant=variant)
    # Invalid filter values.
    except ValueError as exc:
        # Report and exit.
        fail(str(exc))
    # JSON output.
    if as_json:
        # One dictionary per model.
        print_json([e.to_dict() for e in entries])
        # Done.
        return
    # Table output.
    table = Table(title=f"Model zoo ({len(entries)} models)")
    # Model id column.
    table.add_column("Model ID", style="cyan")
    # Task column.
    table.add_column("Task", style="green")
    # Domain column.
    table.add_column("Domain", style="yellow")
    # Parameter count column.
    table.add_column("Parameters", justify="right")
    # One row per model.
    for e in entries:
        # Row values.
        table.add_row(e.model_id, e.task.value, e.domain, f"{e.num_parameters:,}")
    # Print the table.
    console.print(table)


# Show one model.
@zoo.command("info", help="Show the inputs, outputs and metadata of a model.")
@click.argument("model_id")
def zoo_info(model_id: str) -> None:
    # Imported lazily.
    from unbihexium.zoo import get_model

    # Registry entry.
    entry = get_model(model_id)
    # Unknown models.
    if entry is None:
        # Report and exit.
        fail(f"unknown model {model_id}; see `unbihexium zoo list`")
    # Entry as JSON.
    print_json(entry.to_dict())


# Build or download a model into the store.
@zoo.command("build", help="Build a model into the local store and verify it.")
@click.argument("model_id")
@click.option("--onnx", is_flag=True, help="Also export the model to ONNX.")
@click.option("--force", is_flag=True, help="Rebuild even if the model is cached.")
@click.option("--cache-dir", type=click.Path(), help="Cache root directory.")
def zoo_build(model_id: str, onnx: bool, force: bool, cache_dir: str | None) -> None:
    # Imported lazily.
    from unbihexium.zoo import ensure_model

    # Build, verify and cache the model.
    try:
        # Directory of the cached model.
        directory = ensure_model(model_id, cache_dir=cache_dir, onnx=onnx, force=force)
    # Import errors mean PyTorch is missing.
    except ImportError as exc:
        # Explain the extra to install.
        fail(f"{exc}; {TORCH_HINT}")
    # Unknown models and verification errors.
    except (KeyError, ValueError) as exc:
        # Report and exit.
        fail(str(exc))
    # Report the location.
    console.print(f"[green]Cached:[/] {directory}")


# Compatibility alias of build.
@zoo.command("download", help="Alias of `zoo build`, kept for compatibility.", hidden=True)
@click.argument("model_id")
@click.option("--force", "-f", is_flag=True, help="Rebuild even if the model is cached.")
@click.option("--cache-dir", type=click.Path(), help="Cache root directory.")
@click.pass_context
def zoo_download(ctx: click.Context, model_id: str, force: bool, cache_dir: str | None) -> None:
    # Forward to build.
    ctx.invoke(zoo_build, model_id=model_id, onnx=False, force=force, cache_dir=cache_dir)


# Export a checkpoint to ONNX.
@zoo.command("export", help="Export a model or checkpoint to ONNX and verify it.")
@click.argument("model")
@click.argument("output", type=click.Path())
@click.option("--no-verify", is_flag=True, help="Skip the ONNX Runtime comparison.")
def zoo_export(model: str, output: str, no_verify: bool) -> None:
    # Imported lazily; export needs PyTorch and onnx.
    try:
        # Model loading.
        from unbihexium.zoo import load_model
        from unbihexium.zoo.export import ExportError, export_onnx  # ONNX export.
    # PyTorch or onnx is missing.
    except ImportError as exc:
        # Explain the extra to install.
        fail(f"{exc}; {TORCH_HINT}")
    # Load and export.
    try:
        # Load the model or checkpoint.
        net = load_model(model)
        # Export and verify.
        path = export_onnx(net, output, verify=not no_verify)
    # PyTorch or onnx is missing.
    except ImportError as exc:
        # Explain the extra to install.
        fail(f"{exc}; {TORCH_HINT}")
    # Unknown models, invalid checkpoints and failed comparisons.
    except (KeyError, ValueError, OSError, ExportError) as exc:
        # KeyError messages are quoted by Python; show the plain text.
        fail(str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc))
    # Report the file.
    console.print(f"[green]Exported:[/] {path}")


# Verify a cached model.
@zoo.command("verify", help="Verify the files and weights digest of a cached model.")
@click.argument("model_id")
@click.option("--cache-dir", type=click.Path(), help="Cache root directory.")
def zoo_verify(model_id: str, cache_dir: str | None) -> None:
    # Imported lazily.
    from unbihexium.zoo import verify_model

    # Check the checksums of every file and the digest.
    if verify_model(model_id, cache_dir=cache_dir):
        # Success.
        console.print(f"[green]Verified:[/] {model_id}")
    # Missing or modified files.
    else:
        # Report and exit.
        fail(f"{model_id} is not cached or does not verify")


# Location of a cached model.
@zoo.command("where", help="Print the checkpoint path of a cached model.")
@click.argument("model_id")
@click.option("--cache-dir", type=click.Path(), help="Cache root directory.")
def zoo_where(model_id: str, cache_dir: str | None) -> None:
    # Imported lazily.
    from unbihexium.zoo import get_cached_model_path

    # Checkpoint path, or None; invalid ids are errors.
    try:
        # Look up the store.
        path = get_cached_model_path(model_id, cache_dir)
    # Ids with path separators.
    except ValueError as exc:
        # Report and exit.
        fail(str(exc))
    # Models that are not cached.
    if path is None:
        # Report and exit.
        fail(f"{model_id} is not cached; run `unbihexium zoo build {model_id}`")
    # Print the path.
    click.echo(str(path))


# Remove cached models.
@zoo.command("clear", help="Remove one or all models from the local store.")
@click.argument("model_id", required=False)
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
@click.option("--cache-dir", type=click.Path(), help="Cache root directory.")
def zoo_clear(model_id: str | None, yes: bool, cache_dir: str | None) -> None:
    # Imported lazily.
    from unbihexium.zoo import clear_cache

    # Ask before removing everything.
    if model_id is None and not yes:
        # Confirmation prompt; aborts on no.
        click.confirm("Remove every cached model?", abort=True)
    # Remove the files; ids outside the store are refused.
    try:
        # Delete the model directories.
        removed = clear_cache(model_id, cache_dir=cache_dir)
    # Invalid or unknown ids.
    except ValueError as exc:
        # Report and exit.
        fail(str(exc))
    # Report the number of removed models.
    console.print(f"Removed {removed} model(s)")


# Train a model.
@main.command(help="Train or fine-tune a model zoo model.")
@click.argument("model")
@click.option("--data", type=click.Path(exists=True, file_okay=False), help="Dataset root.")
@click.option("--synthetic", type=int, help="Train on this many synthetic samples instead.")
@click.option("--variant", help="Variant for family names: tiny, base, large or mega.")
@click.option("--epochs", default=50, show_default=True, help="Number of epochs.")
@click.option("--batch-size", default=8, show_default=True, help="Chips per step.")
@click.option("--lr", "learning_rate", default=1e-3, show_default=True, help="Peak learning rate.")
@click.option("--weight-decay", default=1e-4, show_default=True, help="AdamW weight decay.")
@click.option("--chip-size", type=int, help="Chip size in pixels; default is the tile size.")
@click.option("--samples-per-epoch", type=int, help="Random chips per epoch.")
@click.option("--device", default="auto", show_default=True, help="auto, cpu, cuda or mps.")
@click.option("--workers", default=0, show_default=True, help="Data loader processes.")
@click.option("--seed", default=0, show_default=True, help="Random seed.")
@click.option("--amp", is_flag=True, help="Mixed precision on CUDA.")
@click.option("--patience", type=int, help="Stop after this many epochs without improvement.")
@click.option(
    "--regression-loss",  # Option name.
    type=click.Choice(["l1", "mse", "huber"]),  # Allowed losses.
    default="l1",  # Default loss.
    show_default=True,  # Show the default in the help.
    help="Loss of regression targets.",  # Help text.
)  # End of the option.
@click.option("--no-augment", is_flag=True, help="Disable data augmentation.")
@click.option("--output", default="runs", show_default=True, help="Output directory.")
def train(**options: Any) -> None:
    # Imported lazily; training needs PyTorch.
    try:
        # Training entry point and configuration.
        from unbihexium.ai.training import TrainConfig
        from unbihexium.ai.training import train as run_training  # Training entry point.
    # PyTorch is missing.
    except ImportError as exc:
        # Explain the extra to install.
        fail(f"{exc}; {TORCH_HINT}")
    # Hyperparameters from the options.
    config = TrainConfig(
        epochs=options["epochs"],  # Epochs.
        batch_size=options["batch_size"],  # Batch size.
        learning_rate=options["learning_rate"],  # Learning rate.
        weight_decay=options["weight_decay"],  # Weight decay.
        chip_size=options["chip_size"],  # Chip size.
        samples_per_epoch=options["samples_per_epoch"],  # Chips per epoch.
        device=options["device"],  # Device.
        num_workers=options["workers"],  # Loader processes.
        seed=options["seed"],  # Seed.
        amp=options["amp"],  # Mixed precision.
        patience=options["patience"],  # Early stopping.
        regression_loss=options["regression_loss"],  # Regression loss.
        augment=not options["no_augment"],  # Augmentation.
        output_dir=options["output"],  # Output directory.
    )  # End of the configuration.
    # Data or synthetic samples are required.
    if not options["data"] and not options["synthetic"]:
        # Explain the options.
        fail("pass --data DIR or --synthetic N")
    # Run the training.
    try:
        # History and checkpoints.
        result = run_training(
            options["model"],  # Model.
            options["data"],  # Dataset.
            config,  # Hyperparameters.
            variant=resolve_variant(options["model"], options["variant"]),  # Variant.
            synthetic=options["synthetic"],  # Synthetic samples.
        )  # End of the training.
    # Dataset and configuration problems.
    except (ValueError, KeyError) as exc:
        # Report and exit.
        fail(str(exc))
    # Summary.
    console.print(f"[green]Best epoch:[/] {result.best_epoch}")
    # Best checkpoint.
    console.print(f"[green]Best checkpoint:[/] {result.best_checkpoint}")
    # Metrics of the best checkpoint.
    print_json(result.best_metrics)


# Evaluate a model.
@main.command(help="Evaluate a model on a dataset split.")
@click.argument("model")
@click.option(
    "--data", required=True, type=click.Path(exists=True, file_okay=False), help="Dataset root."
)
@click.option("--split", default="val", show_default=True, help="Split: train, val or test.")
@click.option("--chip-size", type=int, help="Chip size in pixels; default is the tile size.")
@click.option("--batch-size", default=8, show_default=True, help="Chips per forward pass.")
@click.option("--device", default="auto", show_default=True, help="auto, cpu, cuda or mps.")
@click.option("--threshold", default=0.3, show_default=True, help="Detection score threshold.")
def evaluate(
    model: str,  # Model, model id or checkpoint.
    data: str,  # Dataset root.
    split: str,  # Split.
    chip_size: int | None,  # Chip size.
    batch_size: int,  # Batch size.
    device: str,  # Device.
    threshold: float,  # Detection threshold.
) -> None:  # The command returns nothing.
    # Imported lazily; evaluation needs PyTorch.
    try:
        # Evaluation entry point.
        from unbihexium.ai.training import evaluate as run_evaluation
    # PyTorch is missing.
    except ImportError as exc:
        # Explain the extra to install.
        fail(f"{exc}; {TORCH_HINT}")

    # Metrics of the model.
    try:
        # Evaluate on the split.
        metrics = run_evaluation(model, data, split, chip_size, batch_size, device, threshold)
    # Dataset problems.
    except (ValueError, KeyError) as exc:
        # Report and exit.
        fail(str(exc))
    # Print the metrics.
    print_json(metrics)


# Run a model on a raster.
@main.command(help="Run a model on a raster and write the result.")
@click.argument("model")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option(
    "--second",
    type=click.Path(exists=True, dir_okay=False),  # Existing file.
    help="Second date for change detection.",  # Help text.
)
@click.option("--variant", help="Variant for family names.")
@click.option("--threshold", type=float, help="Detection or segmentation threshold.")
@click.option("--tile-size", type=int, help="Tile size in pixels.")
@click.option("--overlap", default=0.25, show_default=True, help="Tile overlap fraction.")
@click.option(
    "--backend",  # Option name.
    type=click.Choice(["auto", "torch", "onnx"]),  # Allowed backends.
    default="auto",  # Default backend.
    show_default=True,  # Show the default in the help.
    help="Inference backend.",  # Help text.
)  # End of the option.
@click.option("--device", default="cpu", show_default=True, help="Torch device.")
def predict(
    model: str,  # Model, model id, checkpoint or ONNX file.
    input_path: str,  # Input raster.
    output_path: str,  # Output file.
    second: str | None,  # Second date.
    variant: str | None,  # Variant.
    threshold: float | None,  # Threshold.
    tile_size: int | None,  # Tile size.
    overlap: float,  # Overlap.
    backend: str,  # Backend.
    device: str,  # Device.
) -> None:  # The command returns nothing.
    # Imported lazily.
    from unbihexium.ai.predict import check_output_path, task_api, write_result

    # Configured default variant for bare family names.
    variant = resolve_variant(model, variant)
    # Inference options.
    options: dict[str, Any] = {
        "variant": variant,  # Variant.
        "tile_size": tile_size,  # Tile size.
        "overlap": overlap,  # Overlap.
        "backend": backend,  # Backend.
        "device": device,  # Device.
    }  # End of the options.
    # Thresholds only when given, so that task defaults apply otherwise.
    if threshold is not None:
        # Add the threshold.
        options["threshold"] = threshold
    # Run the model.
    try:
        # Task API of the model.
        api = task_api(model, **options)
        # The output name must suit the result, before any work is done.
        check_output_path(api.predictor.config.task, output_path)
        # Starter models are flagged.
        warn_if_untrained(model, variant)
        # Change detection with two files.
        if second is not None:
            # Pairwise prediction.
            result = api.predict_pair(input_path, second)  # type: ignore[attr-defined]
        # Single input.
        else:
            # Prediction.
            result = api.predict(input_path)  # type: ignore[attr-defined]
    # PyTorch is missing for a catalogue model or checkpoint.
    except ImportError as exc:
        # Explain the extra to install.
        fail(f"{exc}; {TORCH_HINT}, or pass an ONNX export with the onnx extra")
    # Invalid inputs and models.
    except (ValueError, KeyError, AttributeError) as exc:
        # KeyError messages are quoted by Python; show the plain text.
        fail(str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc))
    # Write the result.
    path = write_result(result, output_path)
    # Report the output.
    console.print(f"[green]Wrote:[/] {path} ({api.model_id})")


# Compatibility alias of predict with the options of earlier releases.
@main.command(help="Alias of `predict`, kept for compatibility.", hidden=True)
@click.argument("model_id")
@click.option("--input", "-i", "input_path", required=True, help="Input file path.")
@click.option("--output", "-o", "output_path", required=True, help="Output file path.")
@click.option("--task", "-t", help="Ignored; the task follows from the model.")
@click.pass_context
def infer(
    ctx: click.Context,  # Click context.
    model_id: str,  # Model.
    input_path: str,  # Input raster.
    output_path: str,  # Output file.
    task: str | None,  # Ignored task name.
) -> None:  # The command returns nothing.
    # Forward to predict with default options.
    ctx.invoke(predict, model=model_id, input_path=input_path, output_path=output_path)


# Pipeline command group.
@main.group(help="Registered processing pipelines.")
def pipeline() -> None:
    # Group without its own behaviour.
    pass


# List pipelines.
@pipeline.command("list", help="List available pipelines.")
@click.option("--domain", "-d", help="Filter by domain.")
def pipeline_list(domain: str | None) -> None:
    # Task APIs register their pipelines on import.
    import unbihexium.ai
    from unbihexium.registry.pipelines import PipelineRegistry  # Pipeline registry.

    # Filtered or complete listing.
    pipelines = PipelineRegistry.by_domain(domain) if domain else PipelineRegistry.list_all()
    # Table output.
    table = Table(title="Pipelines")
    # Id column.
    table.add_column("Pipeline ID", style="cyan")
    # Name column.
    table.add_column("Name", style="green")
    # Domains column.
    table.add_column("Domains", style="yellow")
    # One row per pipeline.
    for p in pipelines:
        # Row values.
        table.add_row(p.pipeline_id, p.name, ", ".join(p.domains))
    # Print the table.
    console.print(table)


# Run a pipeline.
@pipeline.command("run", help="Run a pipeline on raster files and write its result.")
@click.argument("pipeline_id")
@click.option("--input", "-i", "input_path", required=True, help="Input file path.")
@click.option("--input2", "input2_path", help="Second input for two-date pipelines.")
@click.option("--output", "-o", "output_path", required=True, help="Output file path.")
@click.option("--param", "-p", multiple=True, help="Pipeline parameter as KEY=VALUE.")
def pipeline_run(
    pipeline_id: str,  # Registry id.
    input_path: str,  # First input.
    input2_path: str | None,  # Second input.
    output_path: str,  # Output file.
    param: tuple[str, ...],  # Parameters.
) -> None:  # The command returns nothing.
    # Task APIs register their pipelines on import.
    import unbihexium.ai
    from unbihexium.ai.predict import write_result  # Result output.
    from unbihexium.registry.pipelines import PipelineRegistry  # Registry.

    # Parameters as a dictionary; values are parsed as JSON when possible.
    params: dict[str, Any] = {}
    # Parse every parameter.
    for item in param:
        # Split at the first equals sign.
        key, _, value = item.partition("=")
        # JSON values (numbers, booleans) or plain strings.
        try:
            # Parsed value.
            params[key] = json.loads(value)
        # Plain strings.
        except json.JSONDecodeError:
            # Keep the text.
            params[key] = value
    # Unknown pipelines.
    if PipelineRegistry.get(pipeline_id) is None:
        # Report and exit.
        fail(f"pipeline not found: {pipeline_id}; see unbihexium pipeline list")
    # Create the pipeline; unknown or invalid parameters are errors.
    try:
        # Pipeline with its task API.
        created = PipelineRegistry.create(pipeline_id, **params)
    # Unexpected keyword arguments and invalid values.
    except (TypeError, ValueError, KeyError) as exc:
        # Report and exit.
        fail(f"invalid parameters for {pipeline_id}: {exc}")
    # Unknown pipelines.
    if created is None:
        # Report and exit.
        fail(f"pipeline not found: {pipeline_id}")
    # Input files.
    inputs = {"input": input_path, "input1": input_path}
    # Second input.
    if input2_path:
        # Add it.
        inputs["input2"] = input2_path
    # Task API of the pipeline, when it runs a model.
    task = getattr(created, "task", None)
    # Starter models are flagged; checkpoints given as weights are the user's.
    if task is not None:
        # Model and variant of the task API.
        warn_if_untrained(task.source, task.variant)
    # Run the pipeline.
    try:
        # Execute the steps.
        run = created.run(inputs)
    # Pipeline failures.
    except Exception as exc:  # Every failure of a step ends the command.
        # Report and exit.
        fail(str(exc))
    # Result object stored by the task pipelines.
    result = getattr(created, "last_result", None)
    # Write the result when the pipeline produced one.
    if result is not None:
        # Wrong output names are reported without a traceback.
        try:
            # Write the file.
            write_result(result, output_path)
        # Extension that does not suit the result.
        except ValueError as exc:
            # Report and exit.
            fail(str(exc))
    # Report the run.
    console.print(f"[green]Completed:[/] {run.run_id} -> {output_path}")


# Compute a spectral index.
@main.command(help="Compute a spectral index of a raster and write it as GeoTIFF.")
@click.argument("index_name")
@click.option("--input", "-i", "input_path", required=True, help="Input raster file.")
@click.option("--output", "-o", "output_path", required=True, help="Output GeoTIFF file.")
@click.option("--blue", default=2, show_default=True, help="1-based band number of blue.")
@click.option("--green", default=3, show_default=True, help="1-based band number of green.")
@click.option("--red", default=4, show_default=True, help="1-based band number of red.")
@click.option("--nir", default=8, show_default=True, help="1-based band number of near infrared.")
@click.option("--swir1", default=12, show_default=True, help="1-based band number of SWIR 1.6 um.")
@click.option("--swir2", default=13, show_default=True, help="1-based band number of SWIR 2.2 um.")
@click.option("--coastal", default=1, show_default=True, help="1-based band number of coastal.")
@click.option("--rededge1", default=5, show_default=True, help="1-based band number of red edge 1.")
@click.option("--rededge2", default=6, show_default=True, help="1-based band number of red edge 2.")
@click.option("--rededge3", default=7, show_default=True, help="1-based band number of red edge 3.")
@click.option("--nir08", default=9, show_default=True, help="1-based band number of narrow NIR.")
def index(index_name: str, input_path: str, output_path: str, **bands: int) -> None:
    # Imported lazily.
    import numpy as np  # Arrays.

    from unbihexium.core.index import IndexRegistry  # Index formulas.
    from unbihexium.core.raster import Raster  # Raster input and output.

    # Index definition.
    idx = IndexRegistry.get(index_name.upper())
    # Unknown indices.
    if idx is None:
        # Report the available indices and exit.
        fail(f"unknown index {index_name}; available: {', '.join(IndexRegistry.list_all())}")
    # Read the raster.
    raster = Raster.from_file(input_path)
    # Band data.
    data = raster.data
    # Mypy: from_file reads the data.
    assert data is not None
    # Bands required by the formula, by name.
    arrays = {}
    # Collect the required bands.
    for name in idx.bands_required:
        # Bands without an option cannot be mapped.
        if name.lower() not in bands:
            # Explain the problem.
            fail(f"{idx.name} needs band {name}, which has no command line option")
        # 1-based band number of the name.
        number = bands[name.lower()]
        # The band must exist.
        if not 1 <= number <= data.shape[0]:
            # Explain the problem.
            fail(f"{name} is band {number}, but the raster has {data.shape[0]} bands")
        # Band array.
        arrays[name] = data[number - 1]
    # Compute the index.
    values = idx.compute(arrays).astype(np.float32)
    # Georeferenced output raster.
    meta = raster.metadata
    # Same grid as the input.
    crs = meta.crs if meta else "EPSG:4326"
    # Transform of the input.
    transform = meta.transform if meta else None
    # Output raster on that grid.
    out = Raster.from_array(values, crs=crs, transform=transform)
    # Write the GeoTIFF.
    out.to_file(output_path)
    # Report the output.
    console.print(f"[green]Wrote:[/] {output_path} ({idx.name})")


# Start the REST service.
@main.command(help="Start the REST service with uvicorn (needs the serving extra).")
@click.option("--host", help="Listen address; default from the serving configuration.")
@click.option("--port", type=int, help="Port; default from the serving configuration.")
@click.option(
    "--config",  # Option name.
    "config_path",  # Parameter name.
    type=click.Path(exists=True, dir_okay=False),  # An existing YAML file.
    help="YAML configuration file; default UNBIHEXIUM_CONFIG.",  # Help text.
)
@click.option("--proxy-headers", is_flag=True, help="Trust X-Forwarded-* headers from a proxy.")
def serve(host: str | None, port: int | None, config_path: str | None, proxy_headers: bool) -> None:
    # The server and the application need the serving extra.
    try:
        # ASGI server.
        import uvicorn

        # Application factory.
        from unbihexium.serving import create_app
    # Missing optional dependencies.
    except ImportError:
        # Explain how to install them.
        fail("the REST service needs the serving extra: pip install 'unbihexium[serving]'")
    # Environment access.
    import os

    # Layered configuration.
    from unbihexium.config import get_settings, reset_settings

    # A file given on the command line replaces UNBIHEXIUM_CONFIG.
    if config_path is not None:
        # Make every part of the service read the same file.
        os.environ["UNBIHEXIUM_CONFIG"] = str(Path(config_path).resolve())
        # Drop settings cached before.
        reset_settings()
    # Layered settings: defaults, file, environment.
    settings = get_settings()
    # Command line values win over the configuration.
    bind_host = host if host is not None else settings.serving.host
    # Port from the command line or the configuration.
    bind_port = port if port is not None else settings.serving.port
    # Serve until interrupted.
    uvicorn.run(
        create_app(config=settings.serving),  # Application with these settings.
        host=bind_host,  # Listen address.
        port=bind_port,  # Port.
        proxy_headers=proxy_headers,  # Trust forwarded headers.
        log_level=settings.log_level.lower(),  # Same level as the library.
    )


# Name used by earlier releases and the tests.
cli = main

# Run the command line when the module is executed.
if __name__ == "__main__":
    # Entry point.
    main()

# =============================================================================
# End of module src/unbihexium/cli/main.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
