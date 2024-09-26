#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screen_effect;
uniform sampler2D original;
uniform vec2 resolution;

uniform float blur_intensity;
uniform float blur_kernel_size;
uniform float blur_pseudo_overexposure;

uniform float dirt_texture_ratio;
uniform float dirt_texture_rotation;
uniform vec2 dirt_texture_offset;
uniform float dirt_texture_scale;
uniform float dirt_texture_red_scale;
uniform float dirt_texture_green_scale;
uniform float dirt_texture_blue_scale;

void main() {
  float Pi = 6.28318530718; // Pi*2

  // Blur directions: more is better/smoother but slower.
  float Directions = 16.0;
  // Blur quality: more is better/smoother but slower.
  float Quality = 3.0;

  vec2 Radius = blur_kernel_size / resolution.xy;
  vec4 Color = vec4(0.0, 0.0, 0.0, 0.0);
  // Blur calculations
  for (float d = 0.0; d < Pi; d += Pi / Directions) {
    for (float i = 1.0 / Quality; i <= 1.0; i += 1.0 / Quality) {
      Color += texture(original, TexCoord + vec2(cos(d), sin(d)) * Radius * i);
    }
  }
  Color /= Quality * Directions - blur_pseudo_overexposure;
  vec4 blurry_image = Color;

  // Linear interpolation between original image and blurry image using the
  // noise texture as the factor.
  float sin_factor = sin(dirt_texture_rotation);
  float cos_factor = cos(dirt_texture_rotation);
  vec2 dirt_TexCoord =
      dirt_texture_offset +
      dirt_texture_scale * vec2(TexCoord.x, TexCoord.y) *
          mat2(cos_factor, sin_factor, -sin_factor, cos_factor);
  vec4 dirt_texture = texture(screen_effect, dirt_TexCoord);
  float mix_factor = dirt_texture.x;
  vec4 original_image = texture(original, TexCoord);
  vec4 interpolated_img =
      mix(original_image, blurry_image, mix_factor * blur_intensity);

  dirt_texture *= vec4(dirt_texture_red_scale, dirt_texture_green_scale,
                       dirt_texture_blue_scale, 1.0);
  FragColor = mix(interpolated_img, dirt_texture, dirt_texture_ratio);
}