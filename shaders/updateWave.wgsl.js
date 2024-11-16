// exports a string
export default /*wgsl*/ `

// this tells the gpu how to interpret the uniforms it get sent by the cpu (because it came as one big pack of bits)
struct uniforms {
    clickPos: vec2u, //vec2u: a 2d vector of positive integers
    textureSize: vec2u,
    time: f32
}

// telling the gpu what everything it was sent in the bind group is
@group(0) @binding(0) var<uniform> u: uniforms;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var lastTexture: texture_2d<f32>;
@group(0) @binding(3) var beforeLastTexture: texture_2d<f32>;
@group(0) @binding(4) var obstaclesTexture: texture_2d<f32>;
@group(0) @binding(5) var iorTexture: texture_2d<f32>;

const c = 299792458.; //the speed of light
const dt = 0.000000000004; //the time between each frame
const dx = 0.003; //each pixel is 3mm apart
const dy = 0.003;

const pi = 3.141592653589793438;

const wavelength = 0.015; //we want waves with a wavelength of 1.5cm
const frequency = c / wavelength; //so we need to drive a specific frequency to get that

// this function samples a texture but instead of sampling out of bounds, it samples the nearest thing in bounds
// if this was a fragment shader, I could have set up the sampler to do this, but there's no sampler in a compute shader and you have to do it all yourself
fn sampleRebound(texture: texture_2d<f32>, pos: vec2i) -> f32 {
    var samplePos = pos;
    let tsI = vec2i(u.textureSize);
    if (pos.x < 0) {samplePos.x = 0;}
    if (pos.x >= i32(tsI.x)) {samplePos.x = tsI.x - 1;}
    if (pos.y < 0) {samplePos.y = 0;}
    if (pos.y >= i32(tsI.y)) {samplePos.y = tsI.y - 1;}

    return textureLoad(texture, samplePos, 0).r;
}

@compute @workgroup_size(1) fn updateWave( //@workgroup_size(1) means one thread per workgroup (and i already set one workgroup per pixel); because it has the tag @compute, the shader knows to start here
    @builtin(global_invocation_id) id:vec3<u32> //this workgroup knows which one it is
){
    let i = vec2i(id.xy); //because I have each pixel a workgroup, its id corresponds to the pixel it should work on

    let ior = textureLoad(iorTexture, i, 0).r * 2;
    let v = c/ior;

    // if this pixel is an obstacle, make its value the average of wave values around it. this is a way of getting the wave to bounce off the obstacle
    if (textureLoad(obstaclesTexture, i, 0).r == 1.) {
        var obstacleAvg = 0.;

        let rightIsObstacle = sampleRebound(obstaclesTexture, i+vec2i(1, 0));
        let leftIsObstacle = sampleRebound(obstaclesTexture, i+vec2i(-1, 0));
        let topIsObstacle = sampleRebound(obstaclesTexture, i+vec2i(0, 1));
        let bottomIsObstacle = sampleRebound(obstaclesTexture, i+vec2i(0, -1));

        let totalObstacles = rightIsObstacle+leftIsObstacle+topIsObstacle+bottomIsObstacle;

        if (rightIsObstacle == 0.) { obstacleAvg += sampleRebound(lastTexture, i+vec2i(1, 0)); }
        if (leftIsObstacle == 0.) { obstacleAvg += sampleRebound(lastTexture, i+vec2i(-1, 0)); }
        if (topIsObstacle == 0.) { obstacleAvg += sampleRebound(lastTexture, i+vec2i(0, 1)); }
        if (bottomIsObstacle == 0.) { obstacleAvg += sampleRebound(lastTexture, i+vec2i(0, -1)); }

        obstacleAvg /= totalObstacles;

        textureStore(outputTexture, i, vec4f(obstacleAvg));
    }

    //this pixel is the source of the wave, so force its value to follow a sine wave
    else if (i.x==300 && i.y==0) { 
        let t = u.time * dt;
        let theta = u.time * dt * 6.28 * frequency; //gives the wave the wavelength i want
        textureStore(outputTexture, i, vec4f(sin(theta), cos(theta), 0, 0));
    }

    // if it's not an obstacle or the source, we have to figure out what its new value should be while satisfying the wave equation
    else {
        let beforeLastValue = textureLoad(beforeLastTexture, i, 0);
        let lastValue = textureLoad(lastTexture, i, 0);
        let lastValueRight = textureLoad(lastTexture, i + vec2i(1, 0), 0);
        let lastValueLeft = textureLoad(lastTexture, i + vec2i(-1, 0), 0);
        let lastValueTop = textureLoad(lastTexture, i + vec2i(0, 1), 0);
        let lastValueBottom = textureLoad(lastTexture, i + vec2i(0, -1), 0);

        // this is where all the big work of solving the wave equation happens (it's surprisingly short)
        var nextValue = 2*lastValue - beforeLastValue 
        + pow(v*dt/dx, 2)*(lastValueRight - 2*lastValue + lastValueLeft)
        + pow(v*dt/dy, 2)*(lastValueTop - 2*lastValue + lastValueBottom);

        // !might be able to remove the 0.999
        textureStore(outputTexture, i, vec4f(nextValue.r*0.999, nextValue.g*0.999, 0, 0)); //i multiply by a bit less than 1 because it would get too crazy otherwise as the wave rebounds and adds up
    }
}

`

// now on to transcribe.wgsl.js