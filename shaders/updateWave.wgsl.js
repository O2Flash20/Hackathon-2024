export default /*wgsl*/ `

// this tells the gpu how to interpret the uniforms it get sent by the cpu (because it came as one big pack of bits)
struct uniforms {
    time: f32
}

// telling the gpu what everything it was sent in the bind group is
@group(0) @binding(0) var<uniform> u: uniforms;
@group(0) @binding(1) var outputTexture: texture_storage_3d<rg32float, write>;
@group(0) @binding(2) var lastTexture: texture_3d<f32>;
@group(0) @binding(3) var beforeLastTexture: texture_3d<f32>;
@group(0) @binding(4) var obstaclesTexture: texture_2d<f32>;
@group(0) @binding(5) var iorTexture: texture_2d<f32>;
// @group(0) @binding(6) var propTexture: texture_3d<f32>;

_NUMWAVELENGTHS

const c = 299792458.; //the speed of light
const dt = 0.000000000004; //the time between each frame
const dx = 0.003;
const dy = 0.003;
const reflective = false;
const pi = 3.141592653589793438;

@compute @workgroup_size(1) fn updateWave( //@workgroup_size(1) means one thread per workgroup (and i already set one workgroup per pixel); because it has the tag @compute, the shader knows to start here
    @builtin(global_invocation_id) id:vec3<u32> //this workgroup knows which one it is
){
    let i = vec3i(id.xyz); //because I have each pixel a workgroup, its id corresponds to the pixel it should work on

    let ior = textureLoad(iorTexture, i.xy, 0).r * 2;
    let v = c/ior;

    // if this pixel is an obstacle, make its value the average of wave values around it. this is a way of getting the wave to bounce off the obstacle
    if (textureLoad(obstaclesTexture, i.xy, 0).r == 1.) {
        /*var obstacleAvg = 0.;
        let rightIsObstacle = textureLoad(obstaclesTexture, i.xy+vec2i(1, 0), 0).r;
        let leftIsObstacle = textureLoad(obstaclesTexture, i.xy+vec2i(-1, 0), 0).r;
        let topIsObstacle = textureLoad(obstaclesTexture, i.xy+vec2i(0, 1), 0).r;
        let bottomIsObstacle = textureLoad(obstaclesTexture, i.xy+vec2i(0, -1), 0).r;
        //reflective
        let totalObstacles = rightIsObstacle+leftIsObstacle+topIsObstacle+bottomIsObstacle;
        if (rightIsObstacle == 0.) { obstacleAvg += textureLoad(lastTexture, i+vec3i(1, 0, 0), 0).r; }
        if (leftIsObstacle == 0.) { obstacleAvg += textureLoad(lastTexture, i+vec3i(-1, 0, 0), 0).r; }
        if (topIsObstacle == 0.) { obstacleAvg += textureLoad(lastTexture, i+vec3i(0, 1, 0), 0).r; }
        if (bottomIsObstacle == 0.) { obstacleAvg += textureLoad(lastTexture, i+vec3i(0, -1, 0), 0).r; }
        obstacleAvg /= totalObstacles;

        textureStore(outputTexture, i, vec4f(0));*/
    }

    //this pixel is the source of the wave, so force its value to follow a sine wave
    _EMITTER
        let wavelength = ((700-400)*(f32(i.z)+.5)/numWavelengths+400)*0.00009;
        let frequency = c / (wavelength*ior);
        let t = u.time * dt;
        let theta = u.time * dt * 6.28 * frequency; //gives the wave the wavelength i want
        textureStore(outputTexture, i, vec4f(sin(theta), cos(theta), 0, 0));
    }

    // if it's not an obstacle or the source, we have to figure out what its new value should be while satisfying the wave equation
    else {
        let beforeLastValue = textureLoad(beforeLastTexture, i, 0);
        let lastValue = textureLoad(lastTexture, i, 0);

        var lastValueRight = textureLoad(lastTexture, i + vec3i(1, 0, 0), 0);
        var lastValueLeft = textureLoad(lastTexture, i + vec3i(-1, 0, 0), 0);
        var lastValueTop = textureLoad(lastTexture, i + vec3i(0, 1, 0), 0);
        var lastValueBottom = textureLoad(lastTexture, i + vec3i(0, -1, 0), 0);

        if (textureLoad(obstaclesTexture, i.xy+vec2i(1, 0), 0).r == 1 || i.x == 599) {
            if (reflective==true) {lastValueRight = vec4f(0);}
            else {
                // let dirRight = -textureLoad(propTexture, i, 0).rg;
                // var sampleY = dirRight.y * (-1.)/dirRight.x;
                // if (sampleY != sampleY){
                //     sampleY = 0.;
                // }
                // let mixAmount = sampleY%1.;
                // lastValueRight =
                //     mixAmount     * textureLoad(lastTexture, clamp(i + vec3i(-1, i32(floor(sampleY)), 0), vec3i(0, 0, 0), vec3i(599, 599, 100)), 0) + 
                //     (1-mixAmount) * textureLoad(lastTexture, clamp(i + vec3i(-1, i32(ceil(sampleY)), 0), vec3i(0, 0, 0), vec3i(599, 599, 100)), 0);
                lastValueRight = vec4f(lastValue.r, lastValue.g, 0, 0);
            }
        }
        if (textureLoad(obstaclesTexture, i.xy+vec2i(-1, 0), 0).r == 1 || i.x == 0) {
            if (reflective==true) {lastValueLeft = vec4f(0);}
            else {
                lastValueLeft = vec4f(lastValue.r,lastValue.g,0,0);
            }
        }
        if (textureLoad(obstaclesTexture, i.xy+vec2i(0, 1), 0).r == 1 || i.y == 0) {
            if (reflective==true) {lastValueTop = vec4f(0);}
            else {
                lastValueTop = vec4f(lastValue.r,lastValue.g,0,0);
            }
        }
        if (textureLoad(obstaclesTexture, i.xy+vec2i(0, -1), 0).r == 1 || i.y == 599) {
            if (reflective==true) {lastValueBottom = vec4f(0);}
            else {
                lastValueBottom = vec4f(lastValue.r,lastValue.g,0,0);
            }
        }


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