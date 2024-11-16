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
@group(0) @binding(1) var outputTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var lastTexture: texture_2d<f32>;
@group(0) @binding(3) var beforeLastTexture: texture_2d<f32>;
@group(0) @binding(4) var obstaclesTexture: texture_2d<f32>;

const c = 299792458.; //the speed of light
const dt = 0.000000000004; //the time between each frame
const dx = 0.003; //each pixel is 3mm apart
const dy = 0.003;

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
    else if (i.x==300 &&i.y==0) { 
        let t = u.time * dt;
        let theta = u.time * dt * 6.28 * frequency; //gives the wave the wavelength i want
        textureStore(outputTexture, i, vec4f(sin(theta)));
    }

    // if it's not an obstacle or the source, we have to figure out what its new value should be while satisfying the wave equation
    else {
        let beforeLastValue = sampleRebound(beforeLastTexture, i);

        let lastValue = sampleRebound(lastTexture, i);
        let lastValueRight = sampleRebound(lastTexture, i + vec2i(1, 0));
        let lastValueLeft = sampleRebound(lastTexture, i + vec2i(-1, 0));
        let lastValueTop = sampleRebound(lastTexture, i + vec2i(0, 1));
        let lastValueBottom = sampleRebound(lastTexture, i + vec2i(0, -1));

        // this is where all the big work of solving the wave equation happens (it's surprisingly short)
        var nextValue = 2*lastValue - beforeLastValue 
        + pow(c*dt/dx, 2)*(lastValueRight - 2*lastValue + lastValueLeft)
        + pow(c*dt/dy, 2)*(lastValueTop - 2*lastValue + lastValueBottom);

        // if the user clicked and this pixel is in a certain radius of the click, add a peak in the wave
        if (u.clickPos.x >= 0 && u.clickPos.x < u.textureSize.x && u.clickPos.y >= 0 && u.clickPos.y < u.textureSize.y){
            if (distance(vec2f(u.clickPos), vec2f(i)) < 4) {nextValue += 0.001;}
        }

        textureStore(outputTexture, i, vec4f(nextValue*0.999)); //i multiply by a bit less than 1 because it would get too crazy otherwise as the wave rebounds and adds up
    }
}

`

// now on to transcribe.wgsl.js