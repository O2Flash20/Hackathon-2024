export default /*wgsl*/ `

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var propTexture: texture_3d<f32>;

@compute @workgroup_size(1) fn transcribe(
    @builtin(global_invocation_id) id: vec3u
) {
    let i = id.xy;

    let dim = textureDimensions(propTexture);

    var v = vec2f(0);
    for (var j = u32(0); j<dim.z; j++) {
        v += textureLoad(propTexture, vec3u(i, j), 0).rg;
    }
    v /= f32(dim.z);

    textureStore(
        outputTexture, i,
        vec4f(
            v.r,
            v.g,
            -v.r,
            1
        )
    );
}

`