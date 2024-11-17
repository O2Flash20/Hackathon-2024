export default /*wgsl*/ `

@group(0) @binding(0) var outputTexture: texture_storage_3d<rg32float, write>;
@group(0) @binding(1) var thetaTexture: texture_3d<f32>;
@group(0) @binding(2) var obstaclesTexture: texture_2d<f32>;

const pi = 3.1415926535;

fn isObstacle(samplePos: vec2i) -> bool {
    let dim = vec3i(textureDimensions(thetaTexture));
    if (samplePos.x < 0 || samplePos.x >= dim.x || samplePos.y < 0 || samplePos.y >= dim.y) {
        return true;
    }
    else {
        return textureLoad(obstaclesTexture, samplePos, 0).r == 1;
    }
}

@compute @workgroup_size(1) fn getProp(
    @builtin(global_invocation_id) id: vec3u
) {
    let i = vec3i(id);

    if (!isObstacle(i.xy)) {
        let valueThis = textureLoad(thetaTexture, i, 0).r;

        var dx = 0.;
        var dy = 0.;

        let obstacleRight = isObstacle(i.xy+vec2i(1, 0));
        let obstacleLeft = isObstacle(i.xy+vec2i(-1, 0));

        if (obstacleRight) {
            dx = textureLoad(thetaTexture, i + vec3i(-1, 0, 0), 0).r-valueThis;
        }
        else if (obstacleLeft || !(obstacleRight || obstacleLeft) ){
            dx = valueThis-textureLoad(thetaTexture, i + vec3i(1, 0, 0), 0).r ;
        }

        let obstacleUp = isObstacle(i.xy+vec2i(0, 1));
        let obstacleDown = isObstacle(i.xy+vec2i(0, -1));

        if (obstacleUp) {
            dy = valueThis - textureLoad(thetaTexture, i + vec3i(0, -1, 0), 0).r;
        }
        else if (obstacleDown || !(obstacleUp || obstacleDown) ){
            dy = textureLoad(thetaTexture, i + vec3i(0, 1, 0), 0).r - valueThis;
        }

        if (dx > pi) {dx-=2*pi;}
        else if (dx < -pi) {dx += 2*pi;}

        if (dy > pi) {dy -= 2*pi;}
        else if (dy < -pi) {dy += 2*pi;}

        let dir = normalize(vec2f(dx, dy));

        textureStore(outputTexture, id, vec4f(
            dir.x, 
            dir.y, 
            0, 0
        ));
    }
}

`