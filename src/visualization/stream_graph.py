"""Legacy/internal stream-graph visualization helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
import json
from math import cos, pi, sin
from pathlib import Path
from typing import Any

from payment_graph_forecasting.data.stream_graph import STREAM_GRAPH_COLUMNS, StreamGraphDataset

DEFAULT_SAMPLE_EDGES = 5_000
DEFAULT_MAX_RENDER_NODES = 40
DEFAULT_MAX_RENDER_EDGES = 80
DEFAULT_LAYOUT = "flow"
SUPPORTED_LAYOUTS = {"circle", "flow", "spring"}


@dataclass(frozen=True, slots=True)
class StreamGraphVisualizationNode:
    """Serializable node metadata for a renderable stream-graph view."""

    node_id: int
    incident_events: int
    out_events: int
    in_events: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StreamGraphVisualizationEdge:
    """Serializable edge metadata for a renderable stream-graph view."""

    src_idx: int
    dst_idx: int
    event_count: int
    total_btc: float
    total_usd: float
    timestamp_min: int
    timestamp_max: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StreamGraphVisualizationView:
    """Lightweight, honest, renderable projection of a selected stream graph."""

    parquet_path: str
    label: str
    selection_description: str
    source_total_edges: int
    selected_graph_edges_estimate: int | None
    sampled_edges: int
    sampled_nodes: int
    rendered_nodes: int
    rendered_edges: int
    sampled_all_selected_edges: bool
    node_cap_applied: bool
    edge_cap_applied: bool
    sampling_description: str
    projection_description: str
    nodes: tuple[StreamGraphVisualizationNode, ...]
    edges: tuple[StreamGraphVisualizationEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["nodes"] = [node.to_dict() for node in self.nodes]
        payload["edges"] = [edge.to_dict() for edge in self.edges]
        return payload

    def to_text(self) -> str:
        return format_stream_graph_visualization_view(self)


@dataclass(frozen=True, slots=True)
class StreamGraphVisualizationArtifact:
    """Serializable visualization artifact metadata."""

    output_path: str
    output_format: str
    layout: str
    title: str
    view: StreamGraphVisualizationView

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "output_format": self.output_format,
            "layout": self.layout,
            "title": self.title,
            "view": self.view.to_dict(),
        }

    def to_text(self) -> str:
        return format_stream_graph_visualization_artifact(self)


def build_stream_graph_visualization_view(
    dataset: StreamGraphDataset,
    *,
    sample_edges: int | None = DEFAULT_SAMPLE_EDGES,
    max_nodes: int = DEFAULT_MAX_RENDER_NODES,
    max_edges: int = DEFAULT_MAX_RENDER_EDGES,
    batch_size: int = 8_192,
    include_self_loops: bool = False,
) -> StreamGraphVisualizationView:
    """Build a lightweight renderable view over a selected stream graph."""

    if sample_edges is not None and sample_edges <= 0:
        raise ValueError("sample_edges must be positive when provided")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if max_edges <= 0:
        raise ValueError("max_edges must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    descriptor = dataset.describe()
    columns = list(STREAM_GRAPH_COLUMNS)
    incident_events: Counter[int] = Counter()
    out_events: Counter[int] = Counter()
    in_events: Counter[int] = Counter()
    edge_stats: dict[tuple[int, int], dict[str, float | int]] = {}
    sampled_edges = 0

    for batch in dataset.iter_batches(columns=columns, batch_size=batch_size):
        if sample_edges is not None and sampled_edges >= sample_edges:
            break

        frame = batch.to_pandas()
        if frame.empty:
            continue

        if sample_edges is not None:
            remaining = sample_edges - sampled_edges
            if remaining <= 0:
                break
            frame = frame.iloc[:remaining]

        if frame.empty:
            continue

        sampled_edges += int(len(frame))
        src_counts = frame["src_idx"].value_counts(sort=False).to_dict()
        dst_counts = frame["dst_idx"].value_counts(sort=False).to_dict()

        for node_id, count in src_counts.items():
            int_node_id = int(node_id)
            int_count = int(count)
            out_events[int_node_id] += int_count
            incident_events[int_node_id] += int_count
        for node_id, count in dst_counts.items():
            int_node_id = int(node_id)
            int_count = int(count)
            in_events[int_node_id] += int_count
            incident_events[int_node_id] += int_count

        grouped = frame.groupby(["src_idx", "dst_idx"], sort=False).agg(
            event_count=("timestamp", "size"),
            total_btc=("btc", "sum"),
            total_usd=("usd", "sum"),
            timestamp_min=("timestamp", "min"),
            timestamp_max=("timestamp", "max"),
        )
        for (src_idx, dst_idx), row in grouped.iterrows():
            key = (int(src_idx), int(dst_idx))
            current = edge_stats.get(key)
            if current is None:
                edge_stats[key] = {
                    "event_count": int(row["event_count"]),
                    "total_btc": float(row["total_btc"]),
                    "total_usd": float(row["total_usd"]),
                    "timestamp_min": int(row["timestamp_min"]),
                    "timestamp_max": int(row["timestamp_max"]),
                }
                continue
            current["event_count"] = int(current["event_count"]) + int(row["event_count"])
            current["total_btc"] = float(current["total_btc"]) + float(row["total_btc"])
            current["total_usd"] = float(current["total_usd"]) + float(row["total_usd"])
            current["timestamp_min"] = min(int(current["timestamp_min"]), int(row["timestamp_min"]))
            current["timestamp_max"] = max(int(current["timestamp_max"]), int(row["timestamp_max"]))

    ranked_nodes = sorted(incident_events.items(), key=lambda item: (-item[1], item[0]))
    node_cap_applied = len(ranked_nodes) > max_nodes
    rendered_nodes, rendered_edges, edge_cap_applied, projection_description = _build_hub_focus_projection(
        ranked_nodes=ranked_nodes,
        incident_events=incident_events,
        out_events=out_events,
        in_events=in_events,
        edge_stats=edge_stats,
        max_nodes=max_nodes,
        max_edges=max_edges,
        include_self_loops=include_self_loops,
    )

    sampled_all_selected_edges = (
        descriptor.selected_edges_estimate is not None and sampled_edges >= descriptor.selected_edges_estimate
    )
    if sample_edges is None:
        sampled_all_selected_edges = True

    if sample_edges is None:
        sampling_description = f"full selected graph scan ({sampled_edges:,} edges processed)"
    else:
        sampling_description = (
            f"chronological sample of first {sampled_edges:,} selected edges "
            f"(cap={sample_edges:,})"
        )
    return StreamGraphVisualizationView(
        parquet_path=dataset.parquet_path,
        label=dataset.resolved_label,
        selection_description=descriptor.selection.describe(source_total_edges=descriptor.source_total_edges),
        source_total_edges=descriptor.source_total_edges,
        selected_graph_edges_estimate=descriptor.selected_edges_estimate,
        sampled_edges=sampled_edges,
        sampled_nodes=len(incident_events),
        rendered_nodes=len(rendered_nodes),
        rendered_edges=len(rendered_edges),
        sampled_all_selected_edges=sampled_all_selected_edges,
        node_cap_applied=node_cap_applied,
        edge_cap_applied=edge_cap_applied,
        sampling_description=sampling_description,
        projection_description=projection_description,
        nodes=rendered_nodes,
        edges=rendered_edges,
    )


def _build_hub_focus_projection(
    *,
    ranked_nodes: list[tuple[int, int]],
    incident_events: Counter[int],
    out_events: Counter[int],
    in_events: Counter[int],
    edge_stats: dict[tuple[int, int], dict[str, float | int]],
    max_nodes: int,
    max_edges: int,
    include_self_loops: bool,
) -> tuple[
    tuple[StreamGraphVisualizationNode, ...],
    tuple[StreamGraphVisualizationEdge, ...],
    bool,
    str,
]:
    if not ranked_nodes:
        return (), (), False, "empty sampled graph"

    hub_node_id = int(ranked_nodes[0][0])
    if max_nodes == 1:
        hub_node = StreamGraphVisualizationNode(
            node_id=hub_node_id,
            incident_events=int(incident_events[hub_node_id]),
            out_events=int(out_events[hub_node_id]),
            in_events=int(in_events[hub_node_id]),
        )
        return (hub_node,), (), False, f"hub-focus view around node {hub_node_id} with no neighbor slots"

    inbound_edges: list[StreamGraphVisualizationEdge] = []
    outbound_edges: list[StreamGraphVisualizationEdge] = []
    for (src_idx, dst_idx), stats in edge_stats.items():
        if not include_self_loops and src_idx == dst_idx:
            continue
        if dst_idx == hub_node_id and src_idx != hub_node_id:
            inbound_edges.append(
                StreamGraphVisualizationEdge(
                    src_idx=int(src_idx),
                    dst_idx=int(dst_idx),
                    event_count=int(stats["event_count"]),
                    total_btc=float(stats["total_btc"]),
                    total_usd=float(stats["total_usd"]),
                    timestamp_min=int(stats["timestamp_min"]),
                    timestamp_max=int(stats["timestamp_max"]),
                )
            )
        elif src_idx == hub_node_id and dst_idx != hub_node_id:
            outbound_edges.append(
                StreamGraphVisualizationEdge(
                    src_idx=int(src_idx),
                    dst_idx=int(dst_idx),
                    event_count=int(stats["event_count"]),
                    total_btc=float(stats["total_btc"]),
                    total_usd=float(stats["total_usd"]),
                    timestamp_min=int(stats["timestamp_min"]),
                    timestamp_max=int(stats["timestamp_max"]),
                )
            )

    inbound_edges.sort(key=lambda edge: (-edge.event_count, edge.src_idx))
    outbound_edges.sort(key=lambda edge: (-edge.event_count, edge.dst_idx))
    max_neighbor_slots = max(0, max_nodes - 1)
    direction_cap = max(1, min(max_edges // 2, max_neighbor_slots // 2, 10))
    preferred_inbound = min(len(inbound_edges), direction_cap)
    preferred_outbound = min(len(outbound_edges), direction_cap)

    selected_inbound = inbound_edges[:preferred_inbound]
    selected_outbound = outbound_edges[:preferred_outbound]
    remaining_slots = max_neighbor_slots - len(selected_inbound) - len(selected_outbound)

    inbound_cursor = len(selected_inbound)
    outbound_cursor = len(selected_outbound)
    while remaining_slots > 0 and (inbound_cursor < len(inbound_edges) or outbound_cursor < len(outbound_edges)):
        next_inbound = inbound_edges[inbound_cursor] if inbound_cursor < len(inbound_edges) else None
        next_outbound = outbound_edges[outbound_cursor] if outbound_cursor < len(outbound_edges) else None
        choose_inbound = next_outbound is None or (
            next_inbound is not None and next_inbound.event_count >= next_outbound.event_count
        )
        if choose_inbound and next_inbound is not None:
            selected_inbound.append(next_inbound)
            inbound_cursor += 1
        elif next_outbound is not None:
            selected_outbound.append(next_outbound)
            outbound_cursor += 1
        remaining_slots -= 1

    rendered_edges_list = selected_inbound + selected_outbound
    rendered_edges_list.sort(key=lambda edge: (-edge.event_count, edge.src_idx, edge.dst_idx))
    edge_cap_applied = len(inbound_edges) + len(outbound_edges) > len(rendered_edges_list) or len(rendered_edges_list) > max_edges
    rendered_edges = tuple(rendered_edges_list[:max_edges])

    rendered_node_ids = {hub_node_id}
    for edge in rendered_edges:
        rendered_node_ids.add(edge.src_idx)
        rendered_node_ids.add(edge.dst_idx)

    rendered_nodes_ranked = sorted(
        rendered_node_ids,
        key=lambda node_id: (
            0 if node_id == hub_node_id else 1,
            -incident_events[node_id],
            node_id,
        ),
    )
    rendered_nodes = tuple(
        StreamGraphVisualizationNode(
            node_id=int(node_id),
            incident_events=int(incident_events[node_id]),
            out_events=int(out_events[node_id]),
            in_events=int(in_events[node_id]),
        )
        for node_id in rendered_nodes_ranked
    )
    projection_description = (
        f"hub-focus view around node {hub_node_id} with "
        f"{len(selected_inbound)} inbound and {len(selected_outbound)} outbound aggregated links"
    )
    return rendered_nodes, rendered_edges, edge_cap_applied, projection_description


def render_stream_graph_visualization(
    view: StreamGraphVisualizationView,
    output_path: str | Path,
    *,
    layout: str = DEFAULT_LAYOUT,
    title: str | None = None,
    width: int = 1_400,
    height: int = 1_000,
) -> StreamGraphVisualizationArtifact:
    """Render a lightweight stream-graph view to a shareable SVG artifact."""

    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported layout: {layout}")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    destination = Path(output_path)
    if destination.suffix.lower() != ".svg":
        raise ValueError("render_stream_graph_visualization currently supports only .svg output")
    destination.parent.mkdir(parents=True, exist_ok=True)

    resolved_title = title or _build_title_line(view)
    positions = _compute_node_positions(view, layout=layout, width=width, height=height)
    svg = _build_svg_document(
        view,
        positions=positions,
        title=resolved_title,
        width=width,
        height=height,
        layout=layout,
    )
    destination.write_text(svg, encoding="utf-8")
    return StreamGraphVisualizationArtifact(
        output_path=str(destination),
        output_format="svg",
        layout=layout,
        title=resolved_title,
        view=view,
    )


def save_stream_graph_visualization_view(
    view: StreamGraphVisualizationView,
    output_path: str | Path,
) -> str:
    """Save the renderable view as a portable JSON artifact."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(view.to_dict(), indent=2), encoding="utf-8")
    return str(destination)


def save_stream_graph_visualization_artifact(
    artifact: StreamGraphVisualizationArtifact,
    output_path: str | Path,
) -> str:
    """Save artifact metadata as JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
    return str(destination)


def visualize_stream_graph(
    dataset: StreamGraphDataset,
    output_path: str | Path,
    *,
    sample_edges: int | None = DEFAULT_SAMPLE_EDGES,
    max_nodes: int = DEFAULT_MAX_RENDER_NODES,
    max_edges: int = DEFAULT_MAX_RENDER_EDGES,
    batch_size: int = 8_192,
    include_self_loops: bool = False,
    layout: str = DEFAULT_LAYOUT,
    title: str | None = None,
) -> StreamGraphVisualizationArtifact:
    """Build and render a lightweight stream-graph visualization artifact."""

    view = build_stream_graph_visualization_view(
        dataset,
        sample_edges=sample_edges,
        max_nodes=max_nodes,
        max_edges=max_edges,
        batch_size=batch_size,
        include_self_loops=include_self_loops,
    )
    return render_stream_graph_visualization(
        view,
        output_path,
        layout=layout,
        title=title,
    )


def format_stream_graph_visualization_view(view: StreamGraphVisualizationView) -> str:
    """Render human-readable visualization view metadata."""

    lines = [
        "Stream Graph Visualization View",
        f"Label: {view.label}",
        f"Source parquet: {view.parquet_path}",
        f"Selection: {view.selection_description}",
        f"Source edges: {view.source_total_edges:,}",
    ]
    if view.selected_graph_edges_estimate is not None:
        lines.append(f"Selected graph edges: {view.selected_graph_edges_estimate:,}")
    else:
        lines.append("Selected graph edges: unknown (selection requires scan)")
    lines.extend(
        [
            f"Sampled edges: {view.sampled_edges:,}",
            f"Sampled nodes: {view.sampled_nodes:,}",
            f"Rendered nodes: {view.rendered_nodes:,}",
            f"Rendered edges: {view.rendered_edges:,}",
            f"Sampling: {view.sampling_description}",
            f"Projection: {view.projection_description}",
            f"Sampled all selected edges: {view.sampled_all_selected_edges}",
            f"Node cap applied: {view.node_cap_applied}",
            f"Edge cap applied: {view.edge_cap_applied}",
        ]
    )
    return "\n".join(lines)


def format_stream_graph_visualization_artifact(
    artifact: StreamGraphVisualizationArtifact,
) -> str:
    """Render human-readable artifact metadata."""

    return "\n".join(
        [
            "Stream Graph Visualization Artifact",
            f"Output: {artifact.output_path}",
            f"Format: {artifact.output_format}",
            f"Layout: {artifact.layout}",
            f"Title: {artifact.title}",
            "",
            artifact.view.to_text(),
        ]
    )


def _compute_node_positions(
    view: StreamGraphVisualizationView,
    *,
    layout: str,
    width: int,
    height: int,
) -> dict[int, tuple[float, float]]:
    if not view.nodes:
        return {}
    if layout == "flow":
        return _compute_flow_layout(view, width=width, height=height)
    if layout == "circle":
        return _compute_circle_layout(view, width=width, height=height)
    if layout == "spring":
        return _compute_spring_layout(view, width=width, height=height)
    raise ValueError(f"Unsupported layout: {layout}")


def _compute_circle_layout(
    view: StreamGraphVisualizationView,
    *,
    width: int,
    height: int,
) -> dict[int, tuple[float, float]]:
    radius = min(width, height) * 0.32
    center_x = width / 2
    center_y = height / 2 + 30
    total = len(view.nodes)
    return {
        node.node_id: (
            center_x + radius * cos((2 * pi * index) / max(total, 1)),
            center_y + radius * sin((2 * pi * index) / max(total, 1)),
        )
        for index, node in enumerate(view.nodes)
    }


def _compute_flow_layout(
    view: StreamGraphVisualizationView,
    *,
    width: int,
    height: int,
) -> dict[int, tuple[float, float]]:
    if not view.nodes:
        return {}
    positions: dict[int, tuple[float, float]] = {}
    hub_node = view.nodes[0]
    positions[hub_node.node_id] = (width * 0.5, height * 0.55)

    inbound_node_ids = {edge.src_idx for edge in view.edges if edge.dst_idx == hub_node.node_id}
    outbound_node_ids = {edge.dst_idx for edge in view.edges if edge.src_idx == hub_node.node_id}
    mixed_node_ids = (inbound_node_ids & outbound_node_ids) - {hub_node.node_id}
    inbound_only = [
        node for node in view.nodes[1:]
        if node.node_id in inbound_node_ids and node.node_id not in mixed_node_ids
    ]
    outbound_only = [
        node for node in view.nodes[1:]
        if node.node_id in outbound_node_ids and node.node_id not in mixed_node_ids
    ]
    mixed_nodes = [node for node in view.nodes[1:] if node.node_id in mixed_node_ids]

    def _place_vertical(nodes: list[StreamGraphVisualizationNode], x_pos: float, top_y: float, bottom_y: float) -> None:
        total = len(nodes)
        if total == 0:
            return
        if total == 1:
            positions[nodes[0].node_id] = (x_pos, (top_y + bottom_y) / 2)
            return
        step = (bottom_y - top_y) / max(total - 1, 1)
        for index, node in enumerate(nodes):
            positions[node.node_id] = (x_pos, top_y + step * index)

    def _place_horizontal(nodes: list[StreamGraphVisualizationNode], left_x: float, right_x: float, y_pos: float) -> None:
        total = len(nodes)
        if total == 0:
            return
        if total == 1:
            positions[nodes[0].node_id] = ((left_x + right_x) / 2, y_pos)
            return
        step = (right_x - left_x) / max(total - 1, 1)
        for index, node in enumerate(nodes):
            positions[node.node_id] = (left_x + step * index, y_pos)

    _place_vertical(inbound_only, width * 0.2, 320.0, height - 180.0)
    _place_vertical(outbound_only, width * 0.8, 320.0, height - 180.0)
    _place_horizontal(mixed_nodes, width * 0.34, width * 0.66, height - 165.0)
    return positions


def _compute_spring_layout(
    view: StreamGraphVisualizationView,
    *,
    width: int,
    height: int,
) -> dict[int, tuple[float, float]]:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "layout='spring' requires the optional visualization dependency 'networkx'. "
            "Install with: pip install -e '.[viz]'"
        ) from exc

    graph = nx.DiGraph()
    for node in view.nodes:
        graph.add_node(node.node_id)
    for edge in view.edges:
        graph.add_edge(edge.src_idx, edge.dst_idx, weight=edge.event_count)
    raw_positions = nx.spring_layout(graph, seed=42, weight="weight")
    x_values = [float(x_pos) for x_pos, _ in raw_positions.values()]
    y_values = [float(y_pos) for _, y_pos in raw_positions.values()]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    inner_width = width - 220
    inner_height = height - 260

    def _scale(value: float, lower: float, upper: float, span: float, offset: float) -> float:
        if upper == lower:
            return offset + span / 2
        return offset + ((value - lower) / (upper - lower)) * span

    return {
        int(node_id): (
            _scale(float(x_pos), min_x, max_x, inner_width, 110.0),
            _scale(float(y_pos), min_y, max_y, inner_height, 150.0),
        )
        for node_id, (x_pos, y_pos) in raw_positions.items()
    }


def _build_svg_document(
    view: StreamGraphVisualizationView,
    *,
    positions: dict[int, tuple[float, float]],
    title: str,
    width: int,
    height: int,
    layout: str,
) -> str:
    if layout == "flow":
        return _build_hub_focus_svg_document(view, title=title, width=width, height=height)

    max_incident = max((node.incident_events for node in view.nodes), default=1)
    max_edge_count = max((edge.event_count for edge in view.edges), default=1)
    hub_node_id = view.nodes[0].node_id if view.nodes else None
    label_nodes = {hub_node_id} if hub_node_id is not None else set()
    for node in sorted(view.nodes[1:], key=lambda item: (-item.incident_events, item.node_id))[:8]:
        label_nodes.add(node.node_id)
    title_line = _build_title_line(view)
    subtitle_line = _build_subtitle_line(view)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        '<polygon points="0 0, 10 3.5, 0 7" fill="#7a7f87" />',
        "</marker>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#f7f4ee" />',
        f'<rect x="35" y="30" width="{width - 70}" height="{height - 60}" rx="18" fill="#fffdf9" stroke="#e6e0d4" stroke-width="1.5" />',
        f'<text x="70" y="76" font-size="28" font-family="DejaVu Sans, sans-serif" fill="#22303c">{escape(title_line)}</text>',
        f'<text x="70" y="108" font-size="16" font-family="DejaVu Sans, sans-serif" fill="#52606d">{escape(subtitle_line)}</text>',
        _build_summary_panel(view, x_pos=70, y_pos=132),
        _build_legend_panel(x_pos=width - 375, y_pos=132),
    ]

    for edge in view.edges:
        src = positions.get(edge.src_idx)
        dst = positions.get(edge.dst_idx)
        if src is None or dst is None:
            continue
        stroke_width = 1.4 + (4.2 * edge.event_count / max_edge_count)
        stroke_opacity = 0.18 + (0.34 * edge.event_count / max_edge_count)
        lines.append(
            "".join(
                [
                    f'<line x1="{_format_coord(src[0])}" y1="{_format_coord(src[1])}" ',
                    f'x2="{_format_coord(dst[0])}" y2="{_format_coord(dst[1])}" ',
                    f'stroke="#6b7280" stroke-opacity="{stroke_opacity:.2f}" stroke-width="{stroke_width:.2f}" ',
                    'marker-end="url(#arrow)" />',
                ]
            )
        )

    for node in view.nodes:
        x_pos, y_pos = positions.get(node.node_id, (width / 2, height / 2))
        radius = 12.0 + (20.0 * node.incident_events / max_incident)
        is_labeled = node.node_id in label_nodes
        lines.append(
            f'<circle cx="{_format_coord(x_pos)}" cy="{_format_coord(y_pos)}" r="{radius:.2f}" fill="#2f6f8f" fill-opacity="0.9" stroke="#183b56" stroke-width="2" />'
        )
        if is_labeled:
            if node.node_id == hub_node_id:
                label_x = x_pos + radius + 22
                lines.extend(
                    [
                        f'<line x1="{_format_coord(x_pos + radius)}" y1="{_format_coord(y_pos)}" x2="{_format_coord(label_x - 8)}" y2="{_format_coord(y_pos - 8)}" stroke="#94a3b8" stroke-width="1.2" />',
                        f'<text x="{_format_coord(label_x)}" y="{_format_coord(y_pos - 12)}" text-anchor="start" font-size="14" font-family="DejaVu Sans Mono, monospace" fill="#22303c">{node.node_id}</text>',
                        f'<text x="{_format_coord(label_x)}" y="{_format_coord(y_pos + 7)}" text-anchor="start" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#52606d">{node.incident_events} events</text>',
                    ]
                )
            else:
                label_side = -1 if x_pos > width / 2 else 1
                label_x = x_pos + label_side * (radius + 16)
                anchor = "end" if label_side < 0 else "start"
                lines.extend(
                    [
                        f'<line x1="{_format_coord(x_pos + label_side * radius)}" y1="{_format_coord(y_pos)}" x2="{_format_coord(label_x - label_side * 6)}" y2="{_format_coord(y_pos - 6)}" stroke="#94a3b8" stroke-width="1.0" />',
                        f'<text x="{_format_coord(label_x)}" y="{_format_coord(y_pos - 9)}" text-anchor="{anchor}" font-size="12" font-family="DejaVu Sans Mono, monospace" fill="#22303c">{node.node_id}</text>',
                        f'<text x="{_format_coord(label_x)}" y="{_format_coord(y_pos + 7)}" text-anchor="{anchor}" font-size="11" font-family="DejaVu Sans, sans-serif" fill="#52606d">{node.incident_events} events</text>',
                    ]
                )

    lines.extend(
        [
            f'<text x="70" y="{height - 46}" font-size="13" font-family="DejaVu Sans, sans-serif" fill="#66758a">Hub node sits in the center; strongest inbound neighbors are on the left, outbound neighbors on the right.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines)


def _build_hub_focus_svg_document(
    view: StreamGraphVisualizationView,
    *,
    title: str,
    width: int,
    height: int,
) -> str:
    hub_node = view.nodes[0] if view.nodes else None
    inbound_edges = sorted(
        [edge for edge in view.edges if hub_node is not None and edge.dst_idx == hub_node.node_id],
        key=lambda edge: (-edge.event_count, edge.src_idx),
    )
    outbound_edges = sorted(
        [edge for edge in view.edges if hub_node is not None and edge.src_idx == hub_node.node_id],
        key=lambda edge: (-edge.event_count, edge.dst_idx),
    )
    max_edge_count = max((edge.event_count for edge in view.edges), default=1)
    title_line = _build_title_line(view)
    subtitle_line = _build_subtitle_line(view)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f4ee" />',
        f'<rect x="30" y="24" width="{width - 60}" height="{height - 48}" rx="18" fill="#fffdf9" stroke="#e6e0d4" stroke-width="1.5" />',
        f'<text x="54" y="74" font-size="28" font-family="DejaVu Sans, sans-serif" fill="#22303c">{escape(title_line)}</text>',
        f'<text x="54" y="106" font-size="16" font-family="DejaVu Sans, sans-serif" fill="#52606d">{escape(subtitle_line)}</text>',
        _build_summary_panel(view, x_pos=54, y_pos=128),
        _build_legend_panel(x_pos=width - 360, y_pos=128),
    ]

    if hub_node is not None:
        lines.append(_build_hub_card(hub_node, width=width, y_pos=276))
        lines.append(
            f'<text x="{width / 2:.2f}" y="472" text-anchor="middle" font-size="13" font-family="DejaVu Sans, sans-serif" fill="#66758a">The center card is the most active node in the sampled chronological view.</text>'
        )

    lines.append(
        _build_ranked_edge_panel(
            title="Strongest Inbound To Hub",
            subtitle="neighbors sending into the central hub",
            x_pos=54,
            y_pos=520,
            width=520,
            edges=inbound_edges,
            max_edge_count=max_edge_count,
            hub_node_id=hub_node.node_id if hub_node is not None else None,
            inbound=True,
        )
    )
    lines.append(
        _build_ranked_edge_panel(
            title="Strongest Outbound From Hub",
            subtitle="neighbors receiving from the central hub",
            x_pos=width - 574,
            y_pos=520,
            width=520,
            edges=outbound_edges,
            max_edge_count=max_edge_count,
            hub_node_id=hub_node.node_id if hub_node is not None else None,
            inbound=False,
        )
    )
    lines.append(
        f'<text x="54" y="{height - 42}" font-size="13" font-family="DejaVu Sans, sans-serif" fill="#66758a">This is a hub-focused ranked summary, not a full selected-graph drawing.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _build_hub_card(node: StreamGraphVisualizationNode, *, width: int, y_pos: int) -> str:
    x_pos = width / 2 - 150
    return "\n".join(
        [
            f'<rect x="{x_pos:.2f}" y="{y_pos}" width="300" height="150" rx="18" fill="#edf4f8" stroke="#b8ccd8" stroke-width="1.5" />',
            f'<text x="{width / 2:.2f}" y="{y_pos + 34}" text-anchor="middle" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#52606d">CENTRAL HUB</text>',
            f'<text x="{width / 2:.2f}" y="{y_pos + 74}" text-anchor="middle" font-size="24" font-family="DejaVu Sans Mono, monospace" fill="#22303c">{node.node_id}</text>',
            f'<text x="{width / 2:.2f}" y="{y_pos + 106}" text-anchor="middle" font-size="15" font-family="DejaVu Sans, sans-serif" fill="#2f6f8f">incident={node.incident_events}  inbound={node.in_events}  outbound={node.out_events}</text>',
            f'<text x="{width / 2:.2f}" y="{y_pos + 130}" text-anchor="middle" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#66758a">built from the sampled chronological slice, then aggregated by directed pair</text>',
        ]
    )


def _build_ranked_edge_panel(
    *,
    title: str,
    subtitle: str,
    x_pos: int,
    y_pos: int,
    width: int,
    edges: list[StreamGraphVisualizationEdge],
    max_edge_count: int,
    hub_node_id: int | None,
    inbound: bool,
) -> str:
    panel_height = 52 + 34 + max(len(edges), 1) * 32
    lines = [
        f'<rect x="{x_pos}" y="{y_pos}" width="{width}" height="{panel_height}" rx="18" fill="#fffdf9" stroke="#ddd6c8" stroke-width="1.2" />',
        f'<text x="{x_pos + 20}" y="{y_pos + 28}" font-size="18" font-family="DejaVu Sans, sans-serif" fill="#22303c">{escape(title)}</text>',
        f'<text x="{x_pos + 20}" y="{y_pos + 48}" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#66758a">{escape(subtitle)}</text>',
    ]
    if not edges:
        lines.append(
            f'<text x="{x_pos + 20}" y="{y_pos + 88}" font-size="13" font-family="DejaVu Sans, sans-serif" fill="#66758a">No edges in this direction for the current sampled view.</text>'
        )
        return "\n".join(lines)

    bar_x = x_pos + 216
    bar_width = width - 320
    for index, edge in enumerate(edges):
        row_y = y_pos + 82 + index * 32
        neighbor_id = edge.src_idx if inbound else edge.dst_idx
        count = edge.event_count
        fill_width = 0 if max_edge_count <= 0 else bar_width * (count / max_edge_count)
        arrow = "-> hub" if inbound else "hub ->"
        lines.extend(
            [
                f'<text x="{x_pos + 20}" y="{row_y}" font-size="13" font-family="DejaVu Sans Mono, monospace" fill="#22303c">{neighbor_id}</text>',
                f'<text x="{x_pos + 126}" y="{row_y}" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#66758a">{arrow}</text>',
                f'<rect x="{bar_x}" y="{row_y - 12}" width="{bar_width}" height="12" rx="6" fill="#e8edf2" />',
                f'<rect x="{bar_x}" y="{row_y - 12}" width="{fill_width:.2f}" height="12" rx="6" fill="#5b88a8" />',
                f'<text x="{bar_x + bar_width + 14}" y="{row_y}" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#22303c">{count} events</text>',
            ]
        )
    return "\n".join(lines)


def _build_title_line(view: StreamGraphVisualizationView) -> str:
    if view.selection_description == "full graph":
        return f"{view.label}: lightweight full-graph view"
    if len(view.selection_description) <= 84:
        return f"{view.label}: {view.selection_description}"
    return f"{view.label}: selected stream-graph view"


def _build_subtitle_line(view: StreamGraphVisualizationView) -> str:
    return (
        f"Selected graph: {_format_compact_int(view.selected_graph_edges_estimate)} edges"
        if view.selected_graph_edges_estimate is not None
        else "Selected graph: edge count requires a scan"
    )


def _build_summary_panel(view: StreamGraphVisualizationView, *, x_pos: int, y_pos: int) -> str:
    rows = [
        ("Selection", _short_selection_label(view.selection_description)),
        ("Sample", _format_compact_int(view.sampled_edges) + " chronological edges"),
        ("Projection", f"{view.rendered_nodes} nodes / {view.rendered_edges} edges"),
        ("Meaning", view.projection_description),
    ]
    lines = [
        f'<rect x="{x_pos}" y="{y_pos}" width="520" height="114" rx="14" fill="#f6f8fb" stroke="#d8e1ea" stroke-width="1.2" />'
    ]
    for index, (label, value) in enumerate(rows):
        row_y = y_pos + 26 + index * 24
        lines.append(
            f'<text x="{x_pos + 18}" y="{row_y}" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#52606d">{escape(label.upper())}</text>'
        )
        lines.append(
            f'<text x="{x_pos + 130}" y="{row_y}" font-size="14" font-family="DejaVu Sans, sans-serif" fill="#22303c">{escape(value)}</text>'
        )
    return "\n".join(lines)


def _build_legend_panel(*, x_pos: int, y_pos: int) -> str:
    return "\n".join(
        [
            f'<rect x="{x_pos}" y="{y_pos}" width="290" height="114" rx="14" fill="#fffaf0" stroke="#e7dcc1" stroke-width="1.2" />',
            f'<text x="{x_pos + 18}" y="{y_pos + 26}" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#8a6d3b">LEGEND</text>',
            f'<circle cx="{x_pos + 34}" cy="{y_pos + 52}" r="10" fill="#2f6f8f" fill-opacity="0.9" stroke="#183b56" stroke-width="2" />',
            f'<text x="{x_pos + 56}" y="{y_pos + 57}" font-size="13" font-family="DejaVu Sans, sans-serif" fill="#22303c">node size = incident events in sampled view</text>',
            f'<line x1="{x_pos + 24}" y1="{y_pos + 80}" x2="{x_pos + 46}" y2="{y_pos + 80}" stroke="#6b7280" stroke-opacity="0.38" stroke-width="3" marker-end="url(#arrow)" />',
            f'<text x="{x_pos + 56}" y="{y_pos + 85}" font-size="13" font-family="DejaVu Sans, sans-serif" fill="#22303c">edge width = aggregated event count</text>',
            f'<text x="{x_pos + 18}" y="{y_pos + 105}" font-size="12" font-family="DejaVu Sans, sans-serif" fill="#52606d">hub + top neighbors are labeled; left=inbound, right=outbound</text>',
        ]
    )


def _short_selection_label(selection_description: str) -> str:
    if selection_description == "full graph":
        return "full graph"
    if selection_description.startswith("chronological prefix fraction="):
        return selection_description.replace("chronological prefix ", "", 1)
    return selection_description


def _format_compact_int(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value}"


def _format_coord(value: float) -> str:
    return f"{value:.2f}"


__all__ = [
    "DEFAULT_LAYOUT",
    "DEFAULT_MAX_RENDER_EDGES",
    "DEFAULT_MAX_RENDER_NODES",
    "DEFAULT_SAMPLE_EDGES",
    "SUPPORTED_LAYOUTS",
    "StreamGraphVisualizationArtifact",
    "StreamGraphVisualizationEdge",
    "StreamGraphVisualizationNode",
    "StreamGraphVisualizationView",
    "build_stream_graph_visualization_view",
    "format_stream_graph_visualization_artifact",
    "format_stream_graph_visualization_view",
    "render_stream_graph_visualization",
    "save_stream_graph_visualization_artifact",
    "save_stream_graph_visualization_view",
    "visualize_stream_graph",
]
