/* framework header */
#version 430
layout(location = 0) uniform vec4 iResolution;
layout(binding = 0) uniform sampler2D accumulatorTex;


vec3 aces(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// colour grading from tropical trevor's scripts
// https://github.com/trevorvanhoof/ColorGrading
float Luma(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

vec3 tonemap2(vec3 texColor) {
    texColor /= 2.;
   	texColor *= 16.;  // Hardcoded Exposure Adjustment
   	vec3 x = max(vec3(0),texColor-0.004);
   	return (x*(6.2*x+.5))/(x*(6.2*x+1.7)+0.06);
}

void main()
{
	// readback the buffer
	vec4 tex = texelFetch(accumulatorTex,ivec2(gl_FragCoord.xy),0);

	// divide accumulated color by the sample count
	vec3 col = tex.rgb / tex.a;

    // saturation
	float luma = Luma(col);
	col = mix(vec3(luma), col, 1.25);

    vec3 uGain = vec3(4.);
    vec3 uLift = vec3(.002,.00,.005)*1.;
    vec3 uOffset = vec3(.00,.00,.00);
    vec3 uGamma = vec3(-.35);
    col = pow(max(vec3(0.0), col * (1.0 + uGain - uLift) + uLift + uOffset), max(vec3(0.0), 1.0 - uGamma));
    col = pow( col, vec3(1./2.2) );
    col = aces(col);

    // col = pow(col, vec3(1.25)) * 2.5;
    // col = tonemap2(col);

	// present for display
	gl_FragColor = vec4(col,1);
}
