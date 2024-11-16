export default /*wgsl*/ `

struct uniforms {
    brightness: f32
}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var originalTexture: texture_3d<f32>;
@group(0) @binding(2) var<uniform> u: uniforms;

@compute @workgroup_size(1) fn transcribeTexture(
    @builtin(global_invocation_id) id:vec3<u32>
){
    let i = id.xy;

    var v = 0.;
    for (var j = u32(0); j < textureDimensions(originalTexture).z; j++) {
        v += textureLoad(originalTexture, vec3u(i, j), 0).r;
    }
    v /= f32(textureDimensions(originalTexture).z);

    textureStore(
        outputTexture, i, 
        vec4f(
            u.brightness*v, //make positive values red and brighter
            -u.brightness*v, //make negative values green and brighter
            0,
            1
        )
    );
}

`

// on to renderWave.wgsl.js