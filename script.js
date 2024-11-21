// Importing the shader code from the different files. By code, I mean literally a long string containing the code.
// They are really js files containing only the string, the .wgsl is just to tell me it's wgsl (shader) code.
import updateCode from "./shaders/updateWave.wgsl.js"
import colorCode from "./shaders/color.wgsl.js"
import transcribeCode from "./shaders/transcribe.wgsl.js"
import thetaCode from "./shaders/theta.wgsl.js"
import propCode from "./shaders/prop.wgsl.js"
import displayPropCode from "./shaders/displayProp.wgsl.js"
import renderCode from "./shaders/renderWave.wgsl.js"

let shouldStop = false
document.getElementById("startButton").addEventListener("click", function () {
    shouldStop = true
})

async function start(device) {
    let obstacleTexture = await loadTexture(
        document.getElementById("obstacleSelect").value,
        device
    )
    let iorTexture = await loadTexture(
        document.getElementById("IORSelect").value,
        device
    )
    let emitterType = document.getElementById("emitterSelect").value
    let emitterCode = ""
    if (emitterType == "point") {
        emitterCode = `else if (i.x==300 && i.y==1) {`
    }
    else if (emitterType == "direction") {
        emitterCode = `else if (i.y==1) {`
    }
    else if (emitterType == "center") {
        emitterCode = `else if (i.y == 300 && i.x == 300) {`
    }
    let numWavelengths = Number(document.getElementById("numWavelengths").value)
    main({ obstacleTexture, iorTexture, emitterCode, numWavelengths })
}

// a function to load an external image as a texture
async function loadTexture(url, device) {

    async function loadImageBitmap(url) {
        const res = await fetch(url)
        const blob = await res.blob()
        return await createImageBitmap(blob, { colorSpaceConversion: "none" })
    }

    const source = await loadImageBitmap(url)
    const texture = device.createTexture({ // this is the actual texture object that webgpu know what to do with
        label: url,
        format: "rgba8unorm", // <- rgba means red, green, blue, and alpha; 8 means each of those is 8 bits; unorm means unsigned and normalized (values from 0 to 1)
        size: [source.width, source.height],
        usage:
            GPUTextureUsage.TEXTURE_BINDING | //we want to use it as a texture in a bind group
            GPUTextureUsage.COPY_DST | //we want to copy something to it (the next thing we do)
            GPUTextureUsage.RENDER_ATTACHMENT //also need this to copy something to it
    })

    device.queue.copyExternalImageToTexture(
        { source, flipY: false },
        { texture },
        { width: source.width, height: source.height }
    )

    return texture
}

let device; let canvas; let context; let presentationFormat
async function setup() {
    // set up the device (gpu)
    const adapter = await navigator.gpu?.requestAdapter()
    device = await adapter?.requestDevice()
    if (!device) {
        alert("need a browser that supports WebGPU")
        return
    }

    // get the canvas from the html
    canvas = document.getElementById("mainCanvas")
    context = canvas.getContext("webgpu")

    // the gpu prefers a format to use when rendering
    presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    context.configure({
        device,
        format: presentationFormat,
    })

    start(device)
}
setup()

async function main(scene) {
    // Tells the fragment shader how to sample images. This blends pixels linearly and repeats the image for values out of the bounds of [0, 1]
    const linearSampler = device.createSampler({
        addressModeU: "repeat",
        addressModeV: "repeat",
        addressModeW: "repeat",
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
    })

    // -----------------theta setup----------------- //
    const thetaModule = device.createShaderModule({
        code: thetaCode
    })

    const thetaPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: thetaModule }
    })

    const thetaTexture = device.createTexture({
        format: "r32float",
        dimension: "3d",
        size: [canvas.clientWidth, canvas.clientHeight, scene.numWavelengths],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    })

    // -----------------prop(agation) setup-----------------//
    const propModule = device.createShaderModule({
        code: propCode
    })

    const propPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: propModule }
    })

    const propTexture = device.createTexture({
        format: "rg32float",
        dimension: "3d",
        size: [canvas.clientWidth, canvas.clientHeight, scene.numWavelengths],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    })

    const propBindGroup = device.createBindGroup({
        layout: propPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: propTexture.createView() },
            { binding: 1, resource: thetaTexture.createView() },
            { binding: 2, resource: scene.obstacleTexture.createView() }
        ]
    })

    // -----------------update setup----------------- //

    const updateModule = device.createShaderModule({
        label: "wave update shader module",
        code:
            updateCode
                .replace("_NUMWAVELENGTHS", `const numWavelengths = ${scene.numWavelengths};`)
                .replace("_EMITTER", scene.emitterCode)
    })

    const updatePipeline = device.createComputePipeline({ //this is going to be a compute pipeline because i'm not rendering an image, but instead using the gpu to compute a bunch of values (outputting a bunch of floats formatted in a texture because it's easy)
        label: "wave update pipeline",
        layout: "auto",
        compute: { module: updateModule } //use the code for updating
    })

    const obstaclesTexture = scene.obstacleTexture
    const iorTexture = scene.iorTexture

    // to do the simulation (to approximate a second derivative), I need to store 3 frames: this one, the last one, and the before-last one
    // so I have an array of 3 and cycle through them, keeping track of which is the most recent
    let waveTextures = []
    let lastUpdatedTexture = 2 //keeps track of which is the latest
    for (let i = 0; i < 3; i++) {
        waveTextures[i] = device.createTexture({
            label: "wave texture " + i,
            format: "rg32float",
            dimension: "3d",
            size: [canvas.clientWidth, canvas.clientHeight, scene.numWavelengths],
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING // <- storage binding because it's going to be the output of a compute shader (to do that, it needs to be a storage texture), texture binding because it's going to be the input of another compute shader (to do that it needs to be a regular texture)
        })
    }

    // this is going to be a buffer that can be sent right to the gpu so that I can send information from the cpu to the gpu
    // it's called a uniform because it's the same (uniform) for every thread of the gpu
    const updateUniformsBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const updateUniformsValues = new ArrayBuffer(4)
    const updateUniformsViews = {
        time: new Float32Array(updateUniformsValues),
    }

    // setting the initial values of the uniforms
    updateUniformsViews.time[0] = 0

    // -----------------color setup----------------- //
    const colorModule = device.createShaderModule({
        label: "simulation to color module",
        code: colorCode.replace("_NUMWAVELENGTHS", `const numWavelengths = ${scene.numWavelengths};`)
    })

    const colorPipeline = device.createComputePipeline({
        label: "simulation to color pipeline",
        layout: "auto",
        compute: { module: colorModule }
    })

    const colorTexture = device.createTexture({
        label: "wave simulation to color texture",
        format: "rgba8unorm",
        dimension: "2d",
        size: [canvas.clientWidth, canvas.clientHeight],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    })

    const colorUniformsBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const colorUniformsValues = new ArrayBuffer(4)
    const colorUniformsViews = {
        brightness: new Float32Array(colorUniformsValues),
    }
    colorUniformsViews.brightness[0] = 0

    // -----------------transcription setup----------------- //
    //* the wave is in an r32float texture, it needs to be transcribed to an rgba8unorm with a compute shader so that it can be rendered in a fragment shader and shown to the user

    const transcribeModule = device.createShaderModule({
        label: "wave texture transcribe module",
        code: transcribeCode
    })

    const transcribePipeline = device.createComputePipeline({
        label: "wave texture transcribe pipeline",
        layout: "auto",
        compute: { module: transcribeModule }
    })

    const transcribedWaveTexture = device.createTexture({ // <- this texture will be what the user will see
        label: "transcribed wave texture to an rgba8unorm texture",
        format: "rgba8unorm",
        dimension: "2d",
        size: [canvas.clientWidth, canvas.clientHeight],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    })

    // -----------------display prop setup-----------------//
    const displayPropModule = device.createShaderModule({
        code: displayPropCode
    })

    const displayPropPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: displayPropModule }
    })

    const displayPropTexture = device.createTexture({
        format: "rgba8unorm",
        dimension: "2d",
        size: [canvas.clientWidth, canvas.clientHeight],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    })

    const displayPropBindGroup = device.createBindGroup({
        layout: displayPropPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: displayPropTexture.createView() },
            { binding: 1, resource: propTexture.createView() }
        ]
    })

    // -----------------render setup----------------- //
    // to get the texture on the screen, we need a third shader because compute shaders can't write to a canvas

    const renderModule = device.createShaderModule({
        label: "wave render module",
        code: renderCode
    })

    const renderPipeline = device.createRenderPipeline({ //notice it's not a compute pipeline anymore
        label: "wave render pipeline",
        layout: "auto",
        vertex: { //needs vertex information (which will just be two triangles that cover the whole screen)
            module: renderModule
        },
        fragment: { //and fragment information (what to draw on those triangles)
            module: renderModule,
            targets: [{ format: presentationFormat }]
        }
    })

    const renderUniformsBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const renderUniformsValues = new ArrayBuffer(4)
    const renderUniformsViews = {
        renderMode: new Int32Array(renderUniformsValues),
    }
    // setting the initial values of the uniforms
    renderUniformsViews.renderMode[0] = 1

    // for the other shaders, the bind group is set each frame because it changes. this one can just be done once at the beginning
    // this is what the gpu gets sent from the cpu
    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: transcribedWaveTexture.createView() }, //<- the wave texture to render
            { binding: 1, resource: colorTexture.createView() },
            { binding: 2, resource: linearSampler }, //<- a sampler, telling the shader how to sample the texture
            { binding: 3, resource: obstaclesTexture.createView() }, //<- the texture containing the obstacles, because I want to overlay the obstacles on top of the wave
            { binding: 4, resource: iorTexture.createView() },
            { binding: 5, resource: displayPropTexture.createView() },
            { binding: 6, resource: { buffer: renderUniformsBuffer } },
        ]
    })

    const renderPassDescriptor = {
        label: "wave render renderPass",
        colorAttachments: [
            {
                // view: <- to be filled out when we render (it's what we render to)
                clearValue: [0.3, 0.3, 0.3, 1], //these are just kinda defaults that make the render pass act as you would expect
                loadOp: "clear",
                storeOp: "store"
            }
        ]
    }

    let frameCount = 0
    async function render() { // this gets called each frame
        if (shouldStop) {
            shouldStop = false
            await start(device)
            return
        }

        frameCount++
        document.getElementById("frameCount").innerHTML = "Time Elapsed: " + (frameCount * 0.065).toFixed(1) + "fs"

        // -----------------update stuff----------------- //
        updateUniformsViews.time[0] = frameCount
        device.queue.writeBuffer(updateUniformsBuffer, 0, updateUniformsValues) //<- update the buffer object

        const updateBindGroup = device.createBindGroup({
            layout: updatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: updateUniformsBuffer } }, //all the uniforms
                { binding: 1, resource: waveTextures[(lastUpdatedTexture + 1) % 3].createView() }, //the texture we're going to be updating (after this, it will be the most recent)
                { binding: 2, resource: waveTextures[lastUpdatedTexture].createView() }, //the last texture that was updated
                { binding: 3, resource: waveTextures[(lastUpdatedTexture + 2) % 3].createView() }, //the before-last texture
                { binding: 4, resource: obstaclesTexture.createView() }, //the obstacles
                { binding: 5, resource: iorTexture.createView() },
                // { binding: 6, resource: propTexture.createView() }
            ]
        })

        // all this just gets the gpu to do its thing
        const updateEncoder = device.createCommandEncoder({
            label: "wave update command encoder"
        })
        const updateComputePass = updateEncoder.beginComputePass({
            label: "wave update compute pass"
        })
        updateComputePass.setPipeline(updatePipeline)
        updateComputePass.setBindGroup(0, updateBindGroup)
        updateComputePass.dispatchWorkgroups(canvas.clientWidth, canvas.clientHeight, scene.numWavelengths) //make each pixel be dealt with by its own thread (that's what makes the gpu so powerful) (technically a workgroup can be multiple threads, but in gpu code I say it's just one)
        updateComputePass.end()
        const updateCommandBuffer = updateEncoder.finish()
        device.queue.submit([updateCommandBuffer]) //actually makes the gpu run code

        // at this point, the most recent texture has been successfully updated

        lastUpdatedTexture = (lastUpdatedTexture + 1) % 3 //a new texture has just been updated

        // -----------------color stuff-----------------
        colorUniformsViews.brightness[0] = Math.pow(10,document.getElementById("brightness").value)
        device.queue.writeBuffer(colorUniformsBuffer, 0, colorUniformsValues)
        const colorBindGroup = device.createBindGroup({
            layout: colorPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: colorTexture.createView() },
                { binding: 1, resource: waveTextures[lastUpdatedTexture].createView() },
                { binding: 2, resource: { buffer: colorUniformsBuffer } }
            ]
        })

        const colorEncoder = device.createCommandEncoder()
        const colorComputePass = colorEncoder.beginComputePass()
        colorComputePass.setPipeline(colorPipeline)
        colorComputePass.setBindGroup(0, colorBindGroup)
        colorComputePass.dispatchWorkgroups(canvas.clientWidth, canvas.clientHeight)
        colorComputePass.end()
        const colorCommandBuffer = colorEncoder.finish()
        device.queue.submit([colorCommandBuffer])

        // -----------------transcription stuff----------------- //
        const transcribeBindGroup = device.createBindGroup({
            layout: transcribePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: transcribedWaveTexture.createView() },
                { binding: 1, resource: waveTextures[lastUpdatedTexture].createView() },
                { binding: 2, resource: { buffer: colorUniformsBuffer } }
            ]
        })

        const transcribeEncoder = device.createCommandEncoder({
            label: "wave texture transcribe command encoder"
        })
        const transcribeComputePass = transcribeEncoder.beginComputePass({
            label: "wave texture transcribe compute pass"
        })
        transcribeComputePass.setPipeline(transcribePipeline)
        transcribeComputePass.setBindGroup(0, transcribeBindGroup)
        transcribeComputePass.dispatchWorkgroups(canvas.clientWidth, canvas.clientHeight)
        transcribeComputePass.end()
        const transcribeCommandBuffer = transcribeEncoder.finish()
        device.queue.submit([transcribeCommandBuffer])

        // -----------------theta stuff-----------------
        // const thetaBindGroup = device.createBindGroup({
        //     layout: thetaPipeline.getBindGroupLayout(0),
        //     entries: [
        //         { binding: 0, resource: thetaTexture.createView() },
        //         { binding: 1, resource: waveTextures[lastUpdatedTexture].createView() }
        //     ]
        // })

        // const thetaEncoder = device.createCommandEncoder()
        // const thetaComputePass = thetaEncoder.beginComputePass()
        // thetaComputePass.setPipeline(thetaPipeline)
        // thetaComputePass.setBindGroup(0, thetaBindGroup)
        // thetaComputePass.dispatchWorkgroups(canvas.clientWidth, canvas.clientHeight, scene.numWavelengths)
        // thetaComputePass.end()
        // const thetaCommandBuffer = thetaEncoder.finish()
        // device.queue.submit([thetaCommandBuffer])

        // // -----------------prop stuff----------------- //

        // const propEncoder = device.createCommandEncoder()
        // const propComputePass = propEncoder.beginComputePass()
        // propComputePass.setPipeline(propPipeline)
        // propComputePass.setBindGroup(0, propBindGroup)
        // propComputePass.dispatchWorkgroups(canvas.clientWidth, canvas.clientHeight, scene.numWavelengths)
        // propComputePass.end()
        // const propCommandBuffer = propEncoder.finish()
        // device.queue.submit([propCommandBuffer])

        // //  -----------------display prop stuff----------------- //
        // const displayPropEncoder = device.createCommandEncoder()
        // const displayPropComputePass = displayPropEncoder.beginComputePass()
        // displayPropComputePass.setPipeline(displayPropPipeline)
        // displayPropComputePass.setBindGroup(0, displayPropBindGroup)
        // displayPropComputePass.dispatchWorkgroups(canvas.clientWidth, canvas.clientHeight)
        // displayPropComputePass.end()
        // const displayPropCommandBuffer = displayPropEncoder.finish()
        // device.queue.submit([displayPropCommandBuffer])

        // -----------------render stuff----------------- //
        renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView() //set the target of the shader to be the canvas

        const renderMode = document.getElementById("renderSelect").value
        if (renderMode == "wave") {
            renderUniformsViews.renderMode[0] = 0
        }
        else if (renderMode == "color") {
            renderUniformsViews.renderMode[0] = 1
        }
        else if (renderMode == "direction") {
            renderUniformsViews.renderMode[0] = 2
        }
        device.queue.writeBuffer(renderUniformsBuffer, 0, renderUniformsValues)

        const renderEncoder = device.createCommandEncoder({
            label: "wave render command encoder"
        })
        const renderPass = renderEncoder.beginRenderPass(renderPassDescriptor)
        renderPass.setPipeline(renderPipeline)
        renderPass.setBindGroup(0, renderBindGroup)
        renderPass.draw(6)
        renderPass.end()
        const renderCommandBuffer = renderEncoder.finish()
        device.queue.submit([renderCommandBuffer])
        // the wave should now be rendered to the screen

        requestAnimationFrame(render) //how move on to the next frame
    }
    requestAnimationFrame(render)
}