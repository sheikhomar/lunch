jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install jupyterlab-plotly --no-build
jupyter labextension install plotlywidget --no-build
jupyter labextension install @krassowski/jupyterlab_go_to_definition --no-build
jupyter labextension install @jupyterlab/toc --no-build
jupyter labextension install @aquirdturtle/collapsible_headings --no-build

# Build extensions
jupyter lab build

