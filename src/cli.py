from src import __version__

import click

LOGO = rf"""
       ___   ___                          
|\  | |   | |           |   | |\ /| |\ /| 
| + | |-+-  |-+-   -+-  |-+-| | + | | + | 
|  \| |     |           |   | |   | |   | v{__version__}
"""


@click.group(context_settings=dict(help_option_names=["-h", "--help"]), name="NPF-HMM")
@click.version_option(__version__, "--version", "-V", help="Show version and exit")
def cli():  
    """dmps is a CLI for working with DMPS observations data. For more
    information, type ``dmps info``.
    """
    pass


@cli.command()
def info():
    """Get more information about dmps."""
    click.secho(LOGO, fg="green")
    click.echo(
        "\n"
    )
    
    
if __name__ == '__main__':
    cli()
