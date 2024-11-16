export default /*wgsl*/ `

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var waveTexture: texture_3d<f32>;

const numWavelengths = 50;

fn wavelengthToRgb(wavelength: f32) -> vec3f {
    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;

    if (wavelength >= 380.0 && wavelength < 440.0) {
        r = -(wavelength - 440.0) / (440.0 - 380.0);
        g = 0.0;
        b = 1.0;
    } else if (wavelength >= 440.0 && wavelength < 490.0) {
        r = 0.0;
        g = (wavelength - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if (wavelength >= 490.0 && wavelength < 510.0) {
        r = 0.0;
        g = 1.0;
        b = -(wavelength - 510.0) / (510.0 - 490.0);
    } else if (wavelength >= 510.0 && wavelength < 580.0) {
        r = (wavelength - 510.0) / (580.0 - 510.0);
        g = 1.0;
        b = 0.0;
    } else if (wavelength >= 580.0 && wavelength < 645.0) {
        r = 1.0;
        g = -(wavelength - 645.0) / (645.0 - 580.0);
        b = 0.0;
    } else if (wavelength >= 645.0 && wavelength <= 780.0) {
        r = 1.0;
        g = 0.0;
        b = 0.0;
    }

    // Intensity correction for visible spectrum
    var intensity: f32 = 0.0;
    if (wavelength >= 380.0 && wavelength < 420.0) {
        intensity = 0.3 + 0.7 * (wavelength - 380.0) / (420.0 - 380.0);
    } else if (wavelength >= 420.0 && wavelength < 645.0) {
        intensity = 1.0;
    } else if (wavelength >= 645.0 && wavelength <= 780.0) {
        intensity = 0.3 + 0.7 * (780.0 - wavelength) / (780.0 - 645.0);
    }

    return vec3f(r, g, b) * intensity;
}

@compute @workgroup_size(1) fn getIntensity(
    @builtin(global_invocation_id) id:vec3u
) {
    let i = id.xy;
    var col = vec3f(0);
    for (var j = u32(0); j < numWavelengths; j++) {
        let thisWavelength = (700-400)*(f32(j)+.5)/numWavelengths+400;
        let thisColor = wavelengthToRgb(thisWavelength);
        let wave = textureLoad(waveTexture, vec3u(i, j), 0).rg;
        let intensity = wave.r*wave.r+wave.g*wave.g;
        col += 1000 * intensity * thisColor / numWavelengths;
    }

    textureStore(outputTexture, i, vec4f(col, 0));
}

`