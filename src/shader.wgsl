// Audio visualizer bar shader
// Each bar is an instanced quad (6 vertices = 2 triangles)

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct Params {
    num_bars: u32,
};

@group(0) @binding(0) var<storage, read> magnitudes: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let num_bars = f32(params.num_bars);
    let bar_width = 2.0 / num_bars;
    let gap = bar_width * 0.15;
    let actual_width = bar_width - gap;

    // Get magnitude for this bar (clamped to 0..2 range in clip space)
    let height = clamp(magnitudes[instance_index], 0.0, 2.0);

    // Bar X position (left edge, centered in its slot)
    let x_base = -1.0 + f32(instance_index) * bar_width + gap * 0.5;

    // Quad vertices: 2 triangles forming a rectangle
    var local_pos: vec2<f32>;
    switch vertex_index {
        case 0u: { local_pos = vec2<f32>(0.0, 0.0); }            // bottom-left
        case 1u: { local_pos = vec2<f32>(actual_width, 0.0); }    // bottom-right
        case 2u: { local_pos = vec2<f32>(actual_width, height); }  // top-right
        case 3u: { local_pos = vec2<f32>(0.0, 0.0); }            // bottom-left
        case 4u: { local_pos = vec2<f32>(actual_width, height); }  // top-right
        case 5u: { local_pos = vec2<f32>(0.0, height); }          // top-left
        default: { local_pos = vec2<f32>(0.0, 0.0); }
    }

    // Position in clip space: x in [-1, 1], y starts at bottom (-1)
    let x = x_base + local_pos.x;
    let y = -1.0 + local_pos.y;

    // Color gradient based on frequency bin and bar height
    let freq_t = f32(instance_index) / num_bars;      // 0 = low freq, 1 = high freq
    let height_t = local_pos.y / max(height, 0.001);  // 0 = bottom, 1 = top of bar

    // Low frequencies: cyan, mid: green/yellow, high: magenta/pink
    let r = smoothstep(0.3, 0.8, freq_t) + height_t * 0.2;
    let g = 1.0 - abs(freq_t - 0.35) * 2.5;
    let b = (1.0 - smoothstep(0.0, 0.5, freq_t)) + smoothstep(0.7, 1.0, freq_t) * 0.6;

    // Brighten toward top of each bar
    let brightness = 0.5 + 0.5 * height_t;

    var output: VertexOutput;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.color = vec3<f32>(
        clamp(r, 0.0, 1.0),
        clamp(g, 0.0, 1.0),
        clamp(b, 0.0, 1.0),
    ) * brightness;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
