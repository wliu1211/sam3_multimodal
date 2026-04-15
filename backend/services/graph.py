"""
backend/services/graph.py

Data structures for the scene graph.
Every other service imports from here — Node, Edge, SceneGraph are the
shared language between vision.py, sgg.py, vtt.py, renderer.py, and routes.py.
"""

from dataclasses import dataclass, field
from typing import Optional



@dataclass
class Node:
    """
    Represents a single detected instance in the image.

    A node is either a base-class object (a car, a person, a monkey)
    or an attribute instance (a red shirt, a banana, a ball) that
    belongs to a parent base-class node.

    id:           Zero-padded string index e.g. "0000", "0001"
    label:        The concept text e.g. "car", "person", "banana"
    bounding_box: {x1, y1, x2, y2} pixel coordinates
    mask_rle:     Run-length encoded binary mask (matches frontend decoder)
    confidence:   CLIP similarity score in [0, 1]
    is_attribute: True if this node is a sub-part of a parent object
    parent_id:    id of the parent Node if is_attribute is True
    role:         "primary" (user's main focus) or "related" (context object for edges)
    """
    id:           str
    label:        str
    bounding_box: dict                                    
    mask_rle:     list[int]
    confidence:   float
    is_attribute: bool         = False
    parent_id:    Optional[str] = None
    role:         str          = "primary"                          



    @property
    def box_area(self) -> int:
        b = self.bounding_box
        return max(0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))

    @property
    def centroid(self) -> tuple[float, float]:
        b = self.bounding_box
        return ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)

    def distance_to(self, other: "Node") -> float:
        """Euclidean distance between centroids in pixels."""
        cx1, cy1 = self.centroid
        cx2, cy2 = other.centroid
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def box_edge_distance(self, other: "Node") -> float:
        """
        Minimum pixel distance between the nearest edges of two bounding boxes.
        Returns 0.0 when the boxes overlap or share an edge (i.e. truly touching).
        This is stricter than centroid distance for contact-based predicates like
        holding, where the object must be in the subject's hand, not merely nearby.
        """
        a, b = self.bounding_box, other.bounding_box
        gap_x = max(0.0, float(max(a["x1"], b["x1"]) - min(a["x2"], b["x2"])))
        gap_y = max(0.0, float(max(a["y1"], b["y1"]) - min(a["y2"], b["y2"])))
        return (gap_x ** 2 + gap_y ** 2) ** 0.5

    def overlap_ratio(self, other: "Node") -> float:
        """
        Intersection-over-union (IoU) between two bounding boxes.
        Returns 0.0 if there is no overlap.
        """
        a, b = self.bounding_box, other.bounding_box
        ix1 = max(a["x1"], b["x1"])
        iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"])
        iy2 = min(a["y2"], b["y2"])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union        = self.box_area + other.box_area - intersection
        return intersection / union if union > 0 else 0.0

    def vertical_position_relative_to(self, other: "Node") -> float:
        """
        Returns how far above `other` this node is, as a fraction of
        other's height. Positive = this node is above other.
        """
        other_h = other.bounding_box["y2"] - other.bounding_box["y1"]
        if other_h == 0:
            return 0.0
        delta_y = other.centroid[1] - self.centroid[1]
        return delta_y / other_h




@dataclass
class Edge:
    """
    Represents a predicted relationship between two Nodes.

    subject_id:  id of the subject Node  (e.g. "person")
    object_id:   id of the object Node   (e.g. "dog")
    predicate:   relationship label      (e.g. "walking", "holding")
    confidence:  SGG prediction confidence in [0, 1]
    verified:    True if VTT geometric check passed
    downgraded:  True if predicate was lowered from original prediction
    original_predicate: the predicate before downgrading (if downgraded)
    """
    subject_id:          str
    object_id:           str
    predicate:           str
    confidence:          float
    verified:            bool          = False
    downgraded:          bool          = False
    original_predicate:  Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "subject_id":         self.subject_id,
            "object_id":          self.object_id,
            "predicate":          self.predicate,
            "confidence":         round(self.confidence, 4),
            "verified":           self.verified,
            "downgraded":         self.downgraded,
            "original_predicate": self.original_predicate,
        }




@dataclass
class SceneGraph:
    """
    The complete scene graph for one image query.

    nodes: all detected instances (base objects + attribute instances)
    edges: all predicted + verified relationships between nodes
    """
    nodes: list[Node]         = field(default_factory=list)
    edges: list[Edge]         = field(default_factory=list)



    def get_node(self, node_id: str) -> Optional[Node]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Returns all edges where the node is subject or object."""
        return [
            e for e in self.edges
            if e.subject_id == node_id or e.object_id == node_id
        ]

    def base_nodes(self) -> list[Node]:
        """Returns only non-attribute (base class) nodes."""
        return [n for n in self.nodes if not n.is_attribute]

    def attribute_nodes(self) -> list[Node]:
        """Returns only attribute/sub-part nodes."""
        return [n for n in self.nodes if n.is_attribute]

    def verified_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.verified]


    def to_dict(self) -> dict:
        """
        Converts the scene graph to a JSON-serialisable dict
        for inclusion in the API response.
        """
        return {
            "nodes": [
                {
                    "id":           n.id,
                    "label":        n.label,
                    "bounding_box": n.bounding_box,
                    "mask_rle":     n.mask_rle,
                    "confidence":   round(n.confidence, 4),
                    "is_attribute": n.is_attribute,
                    "parent_id":    n.parent_id,
                    "role":         n.role,
                    "centroid":     list(n.centroid),
                }
                for n in self.nodes
            ],
            "edges": [e.to_dict() for e in self.edges],
            "summary": {
                "total_nodes":     len(self.nodes),
                "base_nodes":      len(self.base_nodes()),
                "attribute_nodes": len(self.attribute_nodes()),
                "total_edges":     len(self.edges),
                "verified_edges":  len(self.verified_edges()),
            },
        }