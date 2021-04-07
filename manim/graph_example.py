from manimlib import *

class GraphExample(Scene):
    def construct(self):
        axes = Axes((-5, 5), (-1, 2))
        axes.add_coordinate_labels()

        self.play(Write(axes, lag_ratio=0.01, run_time=1))

        # Axes.get_graph will return the graph of a function
        normal_graph = axes.get_graph(
            lambda x: np.exp(-0.5 * x ** 2),
            color=BLUE,
        )
        
        normal_shifted_graph = axes.get_graph(
            lambda x: np.exp(-0.5 * (x - 1) ** 2),
            color=BLUE,
        )

        normal_scaled_graph = axes.get_graph(
            lambda x: np.exp(-0.5 * (0.5 * (x - 1)) ** 2),
            color=BLUE,
        )

        normal3_graph = axes.get_graph(
            lambda x: np.exp(-0.5 * x ** 2),
            color=BLUE,
        )

        normal_label = axes.get_graph_label(normal_graph, Tex(r"\theta \text{ distribution}"))

        self.play(
            ShowCreation(normal_graph),
            FadeIn(normal_label, RIGHT),
        )
        self.wait()
        self.play(
            ReplacementTransform(normal_graph, normal_shifted_graph),
        )
        self.wait()
        self.play(
            ReplacementTransform(normal_shifted_graph, normal_scaled_graph),
        )
        self.wait(3)
        self.play(
            ReplacementTransform(normal_scaled_graph, normal3_graph),
        )
        self.wait()
        
