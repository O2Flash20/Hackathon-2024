export default /*wgsl*/`

@group(0) @binding(0) var outputTexture: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var waveTexture: texture_3d<f32>;

@compute @workgroup_size(1) fn getAngle(
    @builtin(global_invocation_id) id: vec3u
) {
    let waveValue = textureLoad(waveTexture, id, 0).rg;

    let theta = atan2(waveValue.r, waveValue.g);

    textureStore(
        outputTexture, id, vec4f(theta)
    );
}

`