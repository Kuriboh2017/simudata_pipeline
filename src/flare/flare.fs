/*
Reference to the original lens flare shader file:
https://www.shadertoy.com/view/lsBGDK

Overall, the lens flare is drew in 3 steps:
1. Draw a bright solid circle as the Sun.
2. Draw the diffraction spikes around the Sun. The spikes are randomly genrated
   using an arctangent function and a noise function.
3. Draw the lens flares one by one repeatedly in the opposite direction of the
   Sun. Each flare's exact location and intensity are randomly generated.

Some constant variables are defined in the shader for an aesthetic image. These
variables are not named in the shader.

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
uniform int lens_flares_count;
uniform float lens_flares_seed;
uniform float lens_flares_intensity;

vec4 noise(float p) {
  return texture(noise_texture, vec2(p / resolution.x, .0));
}

vec4 noise(vec2 p) { return texture(noise_texture, p); }

vec4 noise(vec3 p) {
  float m = mod(p.z, 1.0);
  float s = p.z - m;
  float sprev = s - 1.0;
  if (mod(s, 2.0) == 1.0) {
    s--;
    sprev++;
    m = 1.0 - m;
  };
  return mix(texture(noise_texture, p.xy + noise(sprev).yz * 21.421),
             texture(noise_texture, p.xy + noise(s).yz * 14.751), m);
}

vec4 noise(vec4 p) {
  float m = mod(p.w, 1.0);
  float s = p.w - m;
  float sprev = s - 1.0;
  if (mod(s, 2.0) == 1.0) {
    s--;
    sprev++;
    m = 1.0 - m;
  };
  return mix(noise(p.xyz + noise(sprev).wyx * 3531.123420),
             noise(p.xyz + noise(s).wyx * 4521.5314), m);
}

vec4 noise(float p, float lod) {
  return texture(noise_texture, vec2(p / resolution.x, .0), lod);
}

vec4 noise(vec2 p, float lod) { return texture(noise_texture, p, lod); }

vec4 noise(vec3 p, float lod) {
  float m = mod(p.z, 1.0);
  float s = p.z - m;
  float sprev = s - 1.0;
  if (mod(s, 2.0) == 1.0) {
    s--;
    sprev++;
    m = 1.0 - m;
  };

  return mix(texture(noise_texture, p.xy + noise(sprev, lod).yz, lod * 21.421),
             texture(noise_texture, p.xy + noise(s, lod).yz, lod * 14.751), m);
}

vec3 flare(vec2 uv, vec2 sun_uv) {
  vec2 uv_distance = uv - sun_uv;

  // Draw the sun
  vec3 sun = vec3(0.01 + sun_size * .2) / (length(uv_distance));

  // Draw diffraction spikes
  vec3 diffraction_spike =
      sun *
      vec3(
          noise(atan(uv_distance.x, uv_distance.y) * 256.9 + sun_uv.x * 2.0).y *
          .25);
  diffraction_spike *= diffraction_spikes_intensity;

  // Draw lens flare
  vec3 flare = vec3(.0);
  float fltr = length(uv);
  fltr = (fltr * fltr) * .5 + .5;
  fltr = min(fltr, 1.0);
  float i_lens_flares_count = float(lens_flares_count);
  for (float i = 0; i < i_lens_flares_count; i++) {
    vec4 n = noise(lens_flares_seed + i);
    vec4 n2 = noise(lens_flares_seed + i * 2.1);
    vec4 nc = noise(lens_flares_seed + i * 3.3);
    nc += vec4(length(nc));
    nc *= .65;
    for (int i = 0; i < 3; i++) {
      float ip = n.x * 3.0 + float(i) * .1 * n2.y * n2.y * n2.y;
      float is = n.y * n.y * 4.5 * sun_size + .1;
      float ia = (n.z * 4.0 - 2.0) * n2.x * n.y;
      vec2 iuv = (uv * (mix(1.0, length(uv), n.w * n.w))) *
                 mat2(cos(ia), sin(ia), -sin(ia), cos(ia));
      vec2 id = mix(iuv - sun_uv, iuv + sun_uv, ip);
      flare[i] += pow(max(.0, is - (length(id))), .45) / is * .1 * sun_size *
                  nc[i] * fltr;
    }
  }
  flare *= lens_flares_intensity;

  return sun + diffraction_spike + flare;
}

void main() {
  vec2 uv = TexCoord - 0.5;
  uv.x *= resolution.x / resolution.y;
  uv *= 2.0;

  vec2 sun_uv = sun_location;
  sun_uv.x *= 0.7;

  vec3 color = vec3(.0);
  color += flare(uv, sun_uv) * vec3(1.9, 1.9, 2.4);
  color += noise(TexCoord.xy * resolution).xyz * .01;

  vec4 lens_effects = vec4(color, 1.0);
  vec4 base_image = texture(original, TexCoord);
  FragColor = base_image + lens_effects;
}
