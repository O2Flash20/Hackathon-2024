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

    let rightCol = vec3f(1, 0, 0);
    let leftCol = vec3f(0, 1, 0);
    let upCol = vec3f(0, 0, 1);
    let downCol = vec3f(0, 0.9, 0.9);

    let col = clamp(v.r, 0, 1)*rightCol + clamp(-v.r, 0, 1)*leftCol + clamp(v.g, 0, 1)*upCol + clamp(-v.g, 0, 1)*downCol; 

    textureStore(
        outputTexture, i,
        vec4f(
            col,
            1
        )
    );
}

`