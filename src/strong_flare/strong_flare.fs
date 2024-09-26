/*
Reference to the original lens flare shader file:
https://www.shadertoy.com/view/4sX3Rs

Some constant variables are defined in the shader for the aesthetic effects.
These variables are not named in the shader.

*/

#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform vec2 resolution;
uniform sampler2D noise_texture;
uniform sampler2D original;
uniform float sun_size;
uniform vec2 sun_location;
uniform float diffraction_spikes_intensity;
uniform float lens_flares_intensity;
uniform float lens_flares_red_scale;
uniform float lens_flares_green_scale;
uniform float lens_flares_blue_scale;

float noise(float t) {
  return texture(noise_texture, vec2(t, .0) / resolution).x;
}

float noise(vec2 t) { return texture(noise_texture, t / resolution).x; }

vec3 strong_flare(vec2 uv, vec2 sun_uv, float sun_size) {
  vec2 direction = uv - sun_uv;
  vec2 uvd = uv * (length(uv));

  float angle = atan(direction.x, direction.y);
  float distance = length(direction);
  distance = pow(distance, .1);
  float n = noise(vec2(angle * 16.0, distance * 32.0));

  // Draw the Sun
  float sun = sun_size / (length(uv - sun_uv) * 16.0 + 1.0);

  // Draw diffraction spikes
  vec3 diffraction_spike =
      sun *
      vec3(noise(atan(direction.x, direction.y) * 256.9 + sun_uv.x * 2.0)) *
      diffraction_spikes_intensity;

  sun = sun + sun * (sin(noise(sin(angle * 2. + sun_uv.x) * 4.0 -
                               cos(angle * 3. + sun_uv.y)) *
                         16.) *
                         .1 +
                     distance * .1 + .8);

  // Draw lens flares components
  vec3 flares = vec3(.0);
  {
    float f1 = max(0.01 - pow(length(uv + 1.2 * sun_uv), 1.9), .0) * 7.0;

    float f2 =
        max(1.0 / (1.0 + 32.0 * pow(length(uvd + 0.8 * sun_uv), 2.0)), .0) *
        00.25;
    float f22 =
        max(1.0 / (1.0 + 32.0 * pow(length(uvd + 0.85 * sun_uv), 2.0)), .0) *
        00.23;
    float f23 =
        max(1.0 / (1.0 + 32.0 * pow(length(uvd + 0.9 * sun_uv), 2.0)), .0) *
        00.21;

    vec2 uvx = mix(uv, uvd, -0.5);

    float f4 = max(0.01 - pow(length(uvx + 0.4 * sun_uv), 2.4), .0) * 6.0;
    float f42 = max(0.01 - pow(length(uvx + 0.45 * sun_uv), 2.4), .0) * 5.0;
    float f43 = max(0.01 - pow(length(uvx + 0.5 * sun_uv), 2.4), .0) * 3.0;

    uvx = mix(uv, uvd, -.4);

    float f5 = max(0.01 - pow(length(uvx + 0.2 * sun_uv), 5.5), .0) * 2.0;
    float f52 = max(0.01 - pow(length(uvx + 0.4 * sun_uv), 5.5), .0) * 2.0;
    float f53 = max(0.01 - pow(length(uvx + 0.6 * sun_uv), 5.5), .0) * 2.0;

    uvx = mix(uv, uvd, -0.5);

    float f6 = max(0.01 - pow(length(uvx - 0.3 * sun_uv), 1.6), .0) * 6.0;
    float f62 = max(0.01 - pow(length(uvx - 0.325 * sun_uv), 1.6), .0) * 3.0;
    float f63 = max(0.01 - pow(length(uvx - 0.35 * sun_uv), 1.6), .0) * 5.0;

    flares.r += f2 + f4 + f5 + f6;
    flares.g += f22 + f42 + f52 + f62;
    flares.b += f23 + f43 + f53 + f63;
    flares = flares * 1.3 - vec3(length(uvd) * .05);
    flares *= lens_flares_intensity;
  }

  return sun + diffraction_spike + flares;
}

vec3 color_modifier(vec3 color, float factor, float factor2) {
  float w = color.x + color.y + color.z;
  return mix(color, vec3(w) * factor, w * factor2);
}

void main() {
  vec2 uv = TexCoord - 0.5;
  uv.x *= resolution.x / resolution.y;

  vec2 sun_uv = sun_location * .5;
  sun_uv.x *= 0.7;

  vec3 color_scale = vec3(lens_flares_red_scale, lens_flares_green_scale,
                          lens_flares_blue_scale);

  vec3 lens_effects = color_scale * strong_flare(uv, sun_uv, sun_size);
  lens_effects -= noise(TexCoord * resolution) * .015;
  lens_effects = color_modifier(lens_effects, .5, .1);

  vec4 base_image = texture(original, TexCoord);
  FragColor = base_image + vec4(lens_effects, 1.0);
}
