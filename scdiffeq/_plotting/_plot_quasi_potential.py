import plotly.graph_objects as go

def _plot_quasi_potential(self, cmap="viridis", surface_opacity=0.9, cell_color="azure", save_path=False):

    x_mesh, y_mesh = self.DensityDict["x_mesh"], self.DensityDict["y_mesh"]
    quasi_potential = self.quasi_potential

    fig = go.Figure(
        data=[
            go.Surface(
                z=quasi_potential,
                x=x_mesh,
                y=y_mesh,
                opacity=0.9,
                colorbar=dict(lenmode="fraction", len=0.5),
                colorscale=cmap,
            )
        ]
    )
    fig.update_layout(
        title="Four attractor Quasi-Potential",
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=40, r=40, b=40, t=50),
    )

    fig.add_scatter3d(
        x=self.DensityDict["x"],
        y=self.DensityDict["y"],
        z=self.cell_quasi_potential + 0.02,
        mode="markers",
        marker=dict(
            size=2,
            color=cell_color,
            opacity=1,
        ),
    )
    fig.show()
    # save plot
    if save_path:
        fig.write_html(save_path)
    