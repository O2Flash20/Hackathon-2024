export default /*wgsl*/ `

@group(0) @binding(0) var outputTexture: texture_storage_3d<rg32float, write>;
@group(0) @binding(1) var thetaTexture: texture_3d<f32>;

const pi = 3.1415926535;

@compute @workgroup_size(1) fn getProp(
    @builtin(global_invocation_id) id: vec3u
) {
    let i = vec3i(id);

    let valueThis = textureLoad(thetaTexture, i+vec3i(-1, 0, 0), 0).r;
    let valueRight = textureLoad(thetaTexture, i+vec3i(1, 0, 0), 0).r;
    let valueTop = textureLoad(thetaTexture, i+vec3i(0, 1, 0), 0).r;

    var dx = valueRight-valueThis;
    if (dx > pi) {dx-=2*pi;}
    if (dx < -pi) {dx += 2*pi;}

    var dy = valueTop-valueThis;
    if (dy > pi) {dy -= 2*pi;}
    if (dy < -pi) {dy += 2*pi;}

    textureStore(outputTexture, id, vec4f(
        dx,
        dy,
        0, 0
    ));
}

`