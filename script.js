// Importing the shader code from the different files. By code, I mean literally a long string containing the code.
// They are really js files containing only the string, the .wgsl is just to tell me it's wgsl (shader) code.
import updateCode from "./shaders/updateWave.wgsl.js"
import transcribeCode from "./shaders/transcribe.wgsl.js"
import renderCode from "./shaders/renderWave.wgsl.js"

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

async function main() {

    // set up the device (gpu)
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if (!device) {
        alert("need a browser that supports WebGPU")
        return
    }

    // Tells the fragment shader how to sample images. This blends pixels linearly and repeats the image for values out of the bounds of [0, 1]
    const linearSampler = device.createSampler({
        addressModeU: "repeat",
        addressModeV: "repeat",
        addressModeW: "repeat",
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
    })

    // get the canvas from the html
    const canvas = document.getElementById("mainCanvas")
    const context = canvas.getContext("webgpu")

    // the gpu prefers a format to use when rendering
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    context.configure({
        device,
        format: presentationFormat,
    })

    // the simulation also works by putting down peak in the wave where you click, so this gets if the mouse is down and where it is
    let mouseIsDown = false
    canvas.addEventListener("mousedown", function () { mouseIsDown = true })
    canvas.addEventListener("mouseup", function () { mouseIsDown = false })
    let cursorPos = { x: 0, y: 0 }
    document.addEventListener("mousemove", function (e) {
        const canvasBounds = canvas.getBoundingClientRect()
        cursorPos = {
            x: e.clientX - canvasBounds.left,
            y: e.clientY - canvasBounds.top
        }
    })

    // I have 3 separate shaders in this simulation, the first one updates the raw values in the field making the wave

    // -----------------update setup----------------- //

    const updateModule = device.createShaderModule({
        label: "wave update shader module",
        code: updateCode //the actual gpu code
    })

    const updatePipeline = device.createComputePipeline({ //this is going to be a compute pipeline because i'm not rendering an image, but instead using the gpu to compute a bunch of values (outputting a bunch of floats formatted in a texture because it's easy)
        label: "wave update pipeline",
        layout: "auto",
        compute: { module: updateModule } //use the code for updating
    })

    const obstaclesTexture = await loadTexture("diffraction.png", device) //load an image as the obstacles, you can try the other images here

    // to do the simulation (to approximate a second derivative), I need to store 3 frames: this one, the last one, and the before-last one
    // so I have an array of 3 and cycle through them, keeping track of which is the most recent
    let waveTextures = []
    let lastUpdatedTexture = 2 //keeps track of which is the latest
    for (let i = 0; i < 3; i++) {
        waveTextures[i] = device.createTexture({
            label: "wave texture " + i,
            format: "r32float", //r32 means there's only one channel (which is technically red, but that doesn't really mean anything), and the data will be a float because that's the best way to describe the values in the wave
            dimension: "2d",
            size: [canvas.clientWidth, canvas.clientHeight],
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING // <- storage binding because it's going to be the output of a compute shader (to do that, it needs to be a storage texture), texture binding because it's going to be the input of another compute shader (to do that it needs to be a regular texture)
        })
    }

    // this is going to be a buffer that can be sent right to the gpu so that I can send information from the cpu to the gpu
    // it's called a uniform because it's the same (uniform) for every thread of the gpu
    const uniformsBuffer = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const uniformsValues = new ArrayBuffer(24)
    const uniformsViews = {
        clickPos: new Uint32Array(uniformsValues, 0, 2),
        textureSize: new Uint32Array(uniformsValues, 8, 2),
        time: new Float32Array(uniformsValues, 16, 1),
    }

    // setting the initial values of the uniforms
    uniformsViews.clickPos[0] = -1; uniformsViews.clickPos[1] = -1 //<- this is the default value for no click, it will be reset later
    uniformsViews.textureSize[0] = canvas.clientWidth; uniformsViews.textureSize[1] = canvas.clientHeight
    uniformsViews.time[0] = 0

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

    // for the other shaders, the bind group is set each frame because it changes. this one can just be done once at the beginning
    // this is what the gpu gets sent from the cpu
    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: transcribedWaveTexture.createView() }, //<- the wave texture to render
            { binding: 1, resource: linearSampler }, //<- a sampler, telling the shader how to sample the texture
            { binding: 2, resource: obstaclesTexture.createView() } //<- the texture containing the obstacles, because I want to overlay the obstacles on top of the wave
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
    function render() { // this gets called each frame
        frameCount++

        // -----------------update stuff----------------- //
        if (mouseIsDown) {
            uniformsViews.clickPos[0] = cursorPos.x; uniformsViews.clickPos[1] = cursorPos.y
        }
        else {
            uniformsViews.clickPos[0] = -1; uniformsViews.clickPos[1] = -1
        }
        uniformsViews.time[0] = frameCount
        device.queue.writeBuffer(uniformsBuffer, 0, uniformsValues) //<- update the buffer object

        const updateBindGroup = device.createBindGroup({
            layout: updatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformsBuffer } }, //all the uniforms
                { binding: 1, resource: waveTextures[(lastUpdatedTexture + 1) % 3].createView() }, //the texture we're going to be updating (after this, it will be the most recent)
                { binding: 2, resource: waveTextures[lastUpdatedTexture].createView() }, //the last texture that was updated
                { binding: 3, resource: waveTextures[(lastUpdatedTexture + 2) % 3].createView() }, //the before-last texture
                { binding: 4, resource: obstaclesTexture.createView() } //the obstacles
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
        updateComputePass.dispatchWorkgroups(canvas.clientWidth, canvas.clientHeight) //make each pixel be dealt with by its own thread (that's what makes the gpu so powerful) (technically a workgroup can be multiple threads, but in gpu code I say it's just one)
        updateComputePass.end()
        const updateCommandBuffer = updateEncoder.finish()
        device.queue.submit([updateCommandBuffer]) //actually makes the gpu run code

        // at this point, the most recent texture has been successfully updated

        lastUpdatedTexture = (lastUpdatedTexture + 1) % 3 //a new texture has just been updated


        // -----------------transcription stuff----------------- //
        const transcribeBindGroup = device.createBindGroup({
            layout: transcribePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: transcribedWaveTexture.createView() },
                { binding: 1, resource: waveTextures[lastUpdatedTexture].createView() }
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


        // -----------------render stuff----------------- //
        renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView() //set the target of the shader to be the canvas

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
main()

// please move on to see updateWave.wgsl.js