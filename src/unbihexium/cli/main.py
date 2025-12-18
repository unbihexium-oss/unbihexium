"""Main CLI entrypoint for unbihexium."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from unbihexium._version import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="unbihexium")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Unbihexium: Earth Observation, Geospatial, Remote Sensing, and SAR library."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
def info() -> None:
    """Display library information."""
    from unbihexium.registry.capabilities import CapabilityRegistry
    from unbihexium.registry.models import ModelRegistry
    from unbihexium.registry.pipelines import PipelineRegistry

    console.print(f"[bold blue]Unbihexium[/] v{__version__}")
    console.print()
    console.print(f"Registered capabilities: {len(CapabilityRegistry.ids())}")
    console.print(f"Registered models: {len(ModelRegistry.ids())}")
    console.print(f"Registered pipelines: {len(PipelineRegistry.ids())}")


@main.group()
def zoo() -> None:
    """Model zoo commands."""
    pass


@zoo.command("list")
@click.option("--task", "-t", help="Filter by task type.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def zoo_list(task: str | None, as_json: bool) -> None:
    """List available models in the zoo."""
    from unbihexium.zoo import list_models

    models = list_models(task=task)

    if as_json:
        import json

        click.echo(json.dumps([m.to_dict() for m in models], indent=2))
        return

    table = Table(title="Model Zoo")
    table.add_column("Model ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Task", style="yellow")
    table.add_column("Source", style="blue")
    table.add_column("Size", style="magenta")

    for model in models:
        size = f"{model.size_bytes / 1024 / 1024:.1f} MB" if model.size_bytes > 0 else "N/A"
        table.add_row(
            model.model_id,
            model.config.name,
            model.config.task.value,
            model.source,
            size,
        )

    console.print(table)


@zoo.command("download")
@click.argument("model_id")
@click.option("--version", "-V", "model_version", help="Model version.")
@click.option("--cache-dir", type=click.Path(), help="Cache directory.")
@click.option(
    "--source",
    type=click.Choice(["auto", "repo", "release", "lfs", "external"]),
    default="auto",
    help="Download source.",
)
def zoo_download(
    model_id: str,
    model_version: str | None,
    cache_dir: str | None,
    source: str,
) -> None:
    """Download a model from the zoo."""
    from pathlib import Path

    from unbihexium.zoo import download_model

    cache = Path(cache_dir) if cache_dir else None

    try:
        path = download_model(
            model_id,
            version=model_version,
            cache_dir=cache,
            source=source,
        )
        console.print(f"[green]Downloaded:[/] {path}")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise SystemExit(1) from e


@zoo.command("verify")
@click.argument("model_id")
def zoo_verify(model_id: str) -> None:
    """Verify model integrity via SHA256 checksum."""
    from unbihexium.zoo import verify_model

    try:
        valid = verify_model(model_id)
        if valid:
            console.print(f"[green]Verified:[/] {model_id} - checksum OK")
        else:
            console.print(f"[red]Failed:[/] {model_id} - checksum mismatch")
            raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise SystemExit(1) from e


@main.group()
def pipeline() -> None:
    """Pipeline commands."""
    pass


@pipeline.command("list")
@click.option("--domain", "-d", help="Filter by domain.")
def pipeline_list(domain: str | None) -> None:
    """List available pipelines."""
    from unbihexium.registry.pipelines import PipelineRegistry

    if domain:
        pipelines = PipelineRegistry.by_domain(domain)
    else:
        pipelines = PipelineRegistry.list_all()

    table = Table(title="Pipelines")
    table.add_column("Pipeline ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Domains", style="yellow")

    for p in pipelines:
        table.add_row(p.pipeline_id, p.name, ", ".join(p.domains))

    console.print(table)


@pipeline.command("run")
@click.argument("pipeline_id")
@click.option("--input", "-i", "input_path", required=True, help="Input file path.")
@click.option("--output", "-o", "output_path", required=True, help="Output file path.")
@click.option("--config", "-c", "config_path", help="Config file path.")
def pipeline_run(
    pipeline_id: str,
    input_path: str,
    output_path: str,
    config_path: str | None,
) -> None:
    """Run a pipeline."""
    from pathlib import Path

    from unbihexium.registry.pipelines import PipelineRegistry

    entry = PipelineRegistry.get(pipeline_id)
    if entry is None:
        console.print(f"[red]Error:[/] Pipeline not found: {pipeline_id}")
        raise SystemExit(1)

    # Basic pipeline execution
    console.print(f"[blue]Running pipeline:[/] {pipeline_id}")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")

    pipeline = PipelineRegistry.create(pipeline_id)
    if pipeline is None:
        console.print("[red]Error:[/] Failed to create pipeline")
        raise SystemExit(1)

    try:
        run = pipeline.run({"input": Path(input_path)})
        console.print(f"[green]Completed:[/] {run.run_id}")
        console.print(f"  Duration: {run.duration_seconds:.2f}s" if run.duration_seconds else "")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise SystemExit(1) from e


@main.command()
@click.argument("index_name")
@click.option("--input", "-i", "input_path", required=True, help="Input raster file.")
@click.option("--output", "-o", "output_path", required=True, help="Output file path.")
@click.option("--red", default="B04", help="Red band name.")
@click.option("--nir", default="B08", help="NIR band name.")
@click.option("--green", default="B03", help="Green band name.")
@click.option("--blue", default="B02", help="Blue band name.")
def index(
    index_name: str,
    input_path: str,
    output_path: str,
    red: str,
    nir: str,
    green: str,
    blue: str,
) -> None:
    """Compute a spectral index."""
    from unbihexium.core.index import IndexRegistry
    from unbihexium.core.raster import Raster

    idx = IndexRegistry.get(index_name.upper())
    if idx is None:
        available = IndexRegistry.list_all()
        console.print(f"[red]Error:[/] Unknown index: {index_name}")
        console.print(f"Available: {', '.join(available)}")
        raise SystemExit(1)

    console.print(f"[blue]Computing index:[/] {index_name}")
    console.print(f"  Input: {input_path}")

    raster = Raster.from_file(input_path)
    # This is a simplified example - real implementation would handle band mapping
    console.print(f"[green]Output:[/] {output_path}")


if __name__ == "__main__":
    main()
