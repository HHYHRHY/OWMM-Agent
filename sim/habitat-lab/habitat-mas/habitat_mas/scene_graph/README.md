# Scene Graph Python Module

This module provides a set of classes and functions to create and manipulate scene graphs in 3D datasets. A scene graph is a general data structure commonly used in computer graphics for arranging the logical representation of a graphical scene.

## Classes

- [`SceneGraphBase`](./scene_graph_base.py): This is the base class for creating scene graphs. It provides the basic structure and methods for a scene graph.

- [`SceneGraphGibson`](./scene_graph_gt.py): This class extends `SceneGraphBase` and is specifically designed for the Gibson 3D dataset.

- [`SceneGraphRtabmap`](scene_graph_sempcl.py): This class also extends `SceneGraphBase` and is designed for the RTAB-Map 3D dataset.

## Scene Graph Layers
- [`RegionLayer`](./region_layer.py): This class represents a region layer in a scene graph. A region layer is a collection of regions in a scene.
- [`ObjectLayer`](./object_layer.py): This class represents an object layer in a scene graph. An object layer is a collection of objects in a scene.
- [`AgentLayer`](./agent_layer.py): This class represents an agent layer in a scene graph. An agent layer is a collection of agents in a scene.

## Key Methods

- `get_full_graph()`: This method, defined in `SceneGraphBase`, returns the full scene graph.

- `sample_graph(method, *args, **kwargs)`: This method, also defined in `SceneGraphBase`, returns a sub-sampled scene graph. The sub-sampling method and its parameters are passed as arguments.

- `load_gt_scene_graph()`: This method, defined in `SceneGraphGibson`, loads the ground truth scene graph from the Gibson 3D dataset.

## Usage

To use this module, you need to first create an instance of the scene graph class that corresponds to your 3D dataset. For example, if you are working with the Gibson 3D dataset, you would do:

```python
from scene_graph.scene_graph_gt import SceneGraphGibson

scene_graph = SceneGraphGibson(simulator_instance)
```

Then, you can call the methods of the scene graph instance to manipulate the scene graph. For example, to get the full scene graph, you would do:

```python
full_graph = scene_graph.get_full_graph()
```