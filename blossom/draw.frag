/* framework header */
#version 430
layout(location = 0) uniform vec4 iResolution;
layout(location = 1) uniform int iFrame;

 


/* vvv your shader goes here vvv */


#define PI 3.1415926


// Spectrum palette
// IQ https://www.shadertoy.com/view/ll2GD3

vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ) {
    return a + b*cos( 6.28318*(c*t+d) );
}

vec3 spectrum(float n) {
    return pal( n, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
}


//========================================================
// Noise
//========================================================

// https://www.shadertoy.com/view/4djSRW
vec2 hash22(vec2 p)
{
    p += 1.61803398875; // fix artifacts when reseeding
	vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

const uint k = 1103515245U;  // GLIB C

//https://www.shadertoy.com/view/XlXcW4
vec3 hash33( vec3 xs )
{
    uvec3 x = uvec3(xs);
    x = ((x>>8U)^x.yzx)*k;
    x = ((x>>8U)^x.yzx)*k;
    x = ((x>>8U)^x.yzx)*k;
    return vec3(x)*(1.0/float(0xffffffffU));
}

/*
vec3 hash33(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);

}
*/

float hash13(vec3 p3)
{
	return hash33(p3).x;
}

/*
float hash13(vec3 p3)
{
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
*/

vec2 rndcircle(vec2 seed) {
    float a = seed.x * 2. * PI;
    float r = sqrt(seed.y);
    return vec2(r * cos(a), r * sin(a));
}

//========================================================
// Background
//========================================================



vec3 skyTex(vec2 p)
{   

    return vec3(.2);
}



//========================================================
// Skull
//========================================================


bool dbg;

// Big un-optimised distance function, mekes heavy use
// of HG_SDF, smooth min, and IQ's accurate ellipse distance

#define saturate(x) clamp(x, 0., 1.)

float smin(float a, float b, float k){
    float f = clamp(0.5 + 0.5 * ((a - b) / k), 0., 1.);
    return (1. - f) * a + f  * b - f * (1. - f) * k;
}

float smax(float a, float b, float k) {
    return -smin(-a, -b, k);
}

float smin2(float a, float b, float r) {
    vec2 u = max(vec2(r - a,r - b), vec2(0));
    return max(r, min (a, b)) - length(u);
}

float smax2(float a, float b, float r) {
    vec2 u = max(vec2(r + a,r + b), vec2(0));
    return min(-r, max (a, b)) + length(u);
}

float smin3(float a, float b, float k){
    return min(
        smin(a, b, k),
        smin2(a, b, k)
    );
}

float smax3(float a, float b, float k){
    return max(
        smax(a, b, k),
        smax2(a, b, k)
    );
}

void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

// Shortcut for 45-degrees rotation
void pR45(inout vec2 p) {
    p = (p + vec2(p.y, -p.x))*sqrt(0.5);
}

vec3 pRx(vec3 p, float a) {
    pR(p.yz, a); return p;
}

vec3 pRy(vec3 p, float a) {
    pR(p.xz, a); return p;
}

vec3 pRz(vec3 p, float a) {
    pR(p.xy, a); return p;
}



float vmin(vec3 v) {
    return min(min(v.x, v.y), v.z);
}
float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float vmax2(vec2 v) {
    return max(v.x, v.y);
}

float fBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}


float fBox2(vec2 p, vec2 b) {
	vec2 d = abs(p) - b;
	return length(max(d, vec2(0))) + vmax2(min(d, vec2(0)));
}


//========================================================
// Modelling
//========================================================

struct Material {
    vec3 albedo;
    float specular;
    float roughness;
};

struct Model {
    float d;
    vec3 uvw;
    vec3 albedo;
    int id;
};

Material shadeModel(Model model, inout vec3 nor) {
    int id = model.id;
    vec3 p = model.uvw;
    return Material(model.albedo, 0., 0.);
}

float invert;

mat3 lookUp(vec3 up, vec3 forward) {
    vec3 ww = normalize(up);
    vec3 uu = normalize(cross(ww,forward));
    vec3 vv = normalize(cross(uu,ww));
    return mat3(uu, ww, vv);
}

vec3 pLookUp(vec3 p, vec3 up, vec3 forward) {
    return p * lookUp(up, forward);
}



float sdCrystalOne(vec3 size, vec3 p) {
    float d = fBox(p, size);
    d = max(d, -abs(p.x));
    d = max(d, -(d + vmin(size) * .333));
    return d;
}

float sdCrystalLoop(vec3 size, vec3 l, vec3 p, float seed) {

    p.y = max(p.y, .5 * size.y / l.y);
    
    p.y -= size.y * .5;
    size.y *= .5;
    
    vec3 pp = p;
    float d = 1e12;
    
    
    for (int x = 0; x < int(l.x); x++) {
    for (int y = 0; y < int(l.y); y++) {
    for (int z = 0; z < int(l.z); z++) {
        p = pp;
        vec3 c = vec3(x, y, z);
        p -= ((c + .5) / l - .5) * size * 2.;
        vec3 sz = size / l;
        vec3 h3 = hash33(c + 11. + seed);
        //vec3 h3 = hash33(c + 5.);
        //vec3 h3 = hash33(c + 3.);
        p -= (h3 * 2. - 1.) * sz * .5;
        float m = hash13(c * 10. + 27. + seed);
        sz *= mix(.6, 1.5, m);
        sz.xz *= mix(1.8, .45, pow(float(y) / (l.y - 1.), .5));
        float d2 = fBox(p, sz);
        d2 = max(d2, -abs(p.x));
        //d2 = max(d2, -abs(p.z));
        if (h3.z > .5 && c.y > 0.) {
            d2 = max(d2, -abs(p.y - (m * 2. - 1.) * sz.y * .5));
        }
        //d = max(d, -(d2));
        //d2 = max(d2, -(d2 + vmin(sz) * .5));
        d = min(d, d2);
    }
    }
    }
    
    d = max(d, -(d + vmin(size / l) * .5));
    
    return d;
}


float sdCrystalLoop2(vec3 size, vec3 l, vec3 p) {

    size *= .9;
    
    p.y -= size.y * .5;
    size.y *= .5;
    
    vec3 pp = p;
    float d = 1e12;
    
    
    for (int x = 0; x < int(l.x); x++) {
    for (int y = 0; y < int(l.y); y++) {
    for (int z = 0; z < int(l.z); z++) {
        p = pp;
        vec3 c = vec3(x, y, z);
        p -= ((c + .5) / l - .5) * size * 2.;
        vec3 sz = size / l;
        float m = hash13(c+15.);
        sz *= mix(1.1, 1.75, m);
        float d2 = fBox(p, sz) + .01;
        if (c == vec3(0)) {
            d2 = max(d2, -abs(p.x));
        }
        d2 = max(d2, -d);
        d = min(d, d2);
    }
    }
    }
    
    d = max(d, -(d + vmin(size / l) * .5));
    
    
    return d;
}

float sdCrystalField(vec3 p) {
    float d = 1e12;

    float s = .2;

    d = sdCrystalLoop(vec3(.35, 1.6, .35), vec3(2,3,2), pLookUp(p - vec3(.8,0,-.8), vec3(.2,1,-.5), vec3(1,0,1)), 0.);
    d = smin(d, sdCrystalOne(vec3(.13), pLookUp(p - vec3(1.8,-.15,-.3), vec3(0,1,0), vec3(1,0,-.25))), s);
    d = smin(d, sdCrystalLoop2(vec3(.3, .35, .3), vec3(2,1,2), pLookUp(p - vec3(-.3,0,.5), vec3(-.0,1,.2), vec3(.0,0,1)) - vec3(0,-.2,0)), s);

   d = smin(d, sdCrystalLoop(vec3(.15,1.,.15), vec3(1,3,1), pLookUp(p - vec3(-1.8,-.15,-2.3), vec3(-1,2,-.5), vec3(-1,0,-2)), 11.), s);

    return d;
}


Model map(vec3 p) {
    //p.x = -p.x;
    
    // TODO: MOve second outcrop closer

    vec2 im = vec2(.43,.43);
    pR(p.yz, (.5 - im.y) * PI);
    pR(p.xz, (.5 - im.x) * PI * 2.5);
    
    p.y += .6;
    
    p.xz -= vec2(-1,1) * .4;
    


    float ripple = 7.;
    

    //float df = sdCrystalField(vec3(p.x, 0., p.z));
    //df = max(df, 0.);

    
    float d2 = sdCrystalField(p);

    //df = smin(df, .5, .05) - .5;
    
    //df = smin(df, -fract(p.x * ripple)/ripple, .05);
    
    
    float d = p.y + .25;
    
    d = smin(d, length(p - vec3(.6,-2.5,-.7)) - 2.5, .6);
    //d = smin(d, length(p - vec3(.6,-.1,-.7)) - .5, .0);
    d = smin(d, length(p - vec3(-.3,-.5,.5)) - .5, .4);
    //d = smin(d, length(p - vec3(-.3,-.0,.5)) - .5, .0);
    
    float df = pow(d2 + .333, .5) * 1.5;
   // df = d2;
    d += cos(max(df, 0.) * ripple * PI * 2.) * .015;

    //float d = length(p) - .7;
    //d = 1e12;
    //vec3(.6,.57,.55)
    Model m = Model(d, p, vec3(3), 3);

    //d2 = max(d2, -(d - .001));

    //p = abs(p);
    //p -= .5;
    //float d2 = fBox(p, vec3(.3));
    
    
    
    //d2 = max(d2, -(d - .001));

    //m.albedo = fract(d2 * 10.) * vec3(1);
    
    
    //d = d2;
    //d2 = max(d2, -(d - .001));
    
    
    Model m2 = Model(d2 * invert, p, vec3(1), 1);
    
    if (m2.d < m.d) {
        m = m2;
    }
    
    //m = m2;
    
    return m;
}


//========================================================
// Rendering
//========================================================

const float sqrt3 = 1.7320508075688772;


// http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
vec3 calcNormal( in vec3 pos )
{
    vec3 n = vec3(0.0);
    for( int i=0; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+0.001*e).d;
    }
    return normalize(n);
}

vec3 sunPos = normalize(vec3(5,2,2)) * 2.;

/*
vec3 env(vec3 dir) {
    vec3 col = mix(vec3(.5,.7,1) * .05, vec3(.5,.7,1) * 1., smoothstep(-.4, .4, dir.y));
    vec2 pc = vec2(atan(dir.z, dir.x), dir.y) * 30. - 28.96 * 10.;
    vec3 cl = skyTex(pc);
    col *= cl;
    col += pow(cl, vec3(15.)) * 2.;
    return col;
}
*/

struct Hit {
    Model model;
    vec3 pos;
    float len;
};

Hit marchX(vec3 origin, vec3 rayDirection, float maxDist, float understep) {

    vec3 rayPosition;
    float rayLength, dist = 0.;
    Model model;

    for (int i = 0; i < 200; i++) {
        rayPosition = origin + rayDirection * rayLength;
        model = map(rayPosition);
        rayLength += model.d * understep;
        
        if (abs(model.d) / rayLength < .0002) break;

        if (rayLength > maxDist) {
            model.id = 0;
            break;
        }
    }
    return Hit(model, rayPosition, rayLength);
}


Hit march(vec3 origin, vec3 rayDir, float maxDist, float understep) {
    vec3 p;
    float len = 0.;
    float dist = 0.;
    Model model;
    float steps = 0.;

    for (float i = 0.; i < 300.; i++) {
        len += dist;
        p = origin + len * rayDir;
        model = map(p);
        dist = model.d;
        steps += 1.;
        if (abs(model.d) / len < .0002) {
            break;
        }
        if (len >= maxDist) {
            len = maxDist;
            model.id = 0;
            break;
        }
    }   

    return Hit(model, p, len);
}

// tracing/lighting setup from yx
// https://www.shadertoy.com/view/ts2cWm
vec3 ortho(vec3 a){
    vec3 b=cross(vec3(-1,-1,.5),a);
    // assume b is nonzero
    return (b);
}

// re-borrowed from yx from
// http://blog.hvidtfeldts.net/index.php/2015/01/path-tracing-3d-fractals/
vec3 getSampleBiased(vec3  dir, float power, vec2 seed) {
	dir = normalize(dir);
	vec3 o1 = normalize(ortho(dir));
	vec3 o2 = normalize(cross(dir, o1));
	vec2 r = seed;
	r.x=r.x*2.*PI;
	r.y=pow(r.y,1.0/(power+1.0));
	float oneminus = sqrt(1.0-r.y*r.y);
	return cos(r.x)*oneminus*o1+sin(r.x)*oneminus*o2+r.y*dir;
}

vec3 getConeSample(vec3 dir, float extent, vec2 seed) {
	dir = normalize(dir);
	vec3 o1 = normalize(ortho(dir));
	vec3 o2 = normalize(cross(dir, o1));
	vec2 r =  seed;
	r.x=r.x*2.*PI;
	r.y=1.0-r.y*extent;
	float oneminus = sqrt(1.0-r.y*r.y);
	return cos(r.x)*oneminus*o1+sin(r.x)*oneminus*o2+r.y*dir;
}

vec3 BGCOL = vec3(.9,.83,1);




float intersectPlane(vec3 rOrigin, vec3 rayDir, vec3 origin, vec3 normal, vec3 up, out vec2 uv) {
    float d = dot(normal, (origin - rOrigin)) / dot(rayDir, normal);
  	vec3 point = rOrigin + d * rayDir;
	vec3 tangent = cross(normal, up);
	vec3 bitangent = cross(normal, tangent);
    point -= origin;
    uv = vec2(dot(tangent, point), dot(bitangent, point));
    return max(sign(d), 0.);
}

mat3 sphericalMatrix(vec2 tp) {
    float theta = tp.x;
    float phi = tp.y;
    float cx = cos(theta);
    float cy = cos(phi);
    float sx = sin(theta);
    float sy = sin(phi);
    return mat3(
        cy, -sy * -sx, -sy * cx,
        0, cx, sx,
        sy, cy * -sx, cy * cx
    );
}


mat3 envOrientation;

vec3 light(vec3 origin, vec3 rayDir) {

    //vec2 im = iMouse.xy / iResolution.xy;  
    //pR(sunPos.yz, (.5 - im.y) * PI);
    //pR(sunPos.xz, (.5 - im.x) * PI * 2.5);

   // float d = dot(normalize(sunPos - origin), rayDir);
   // return step(.9, d) * vec3(.5);

    origin = -origin;
    rayDir = -rayDir;

    origin *= envOrientation;
    rayDir *= envOrientation;

    vec2 uv;
    vec3 pos = vec3(-6);
    float hit = intersectPlane(origin, rayDir, pos, normalize(pos), normalize(vec3(-1,1,0)), uv);
    float l = smoothstep(.75, .0, fBox2(uv, vec2(.5,2)) - 1.);
    l *= smoothstep(6., 0., length(uv));
	return vec3(l) * hit * 2.;

}


// main path tracing loop, based on yx's
// https://www.shadertoy.com/view/ts2cWm
// with a bit of demofox's
// https://www.shadertoy.com/view/WsBBR3
vec4 draw(vec2 fragCoord) {

    vec2 seed = hash22(fragCoord + (float(iFrame)) * sqrt3);

    invert = 1.;
    
    envOrientation = sphericalMatrix(((vec2(81.5, 119) / vec2(187)) * 2. - 1.) * 2.);

    //vec2 im = iMouse.xy / iResolution.xy;  
    //vec3 v = vec3(0,0,1);
    //pR(v.yz, (.5 - im.y) * PI * 4.);
    //pR(v.xz, (.5 - im.x) * PI * 4.);
    //envOrientation = lookAt(v, vec3(0,1,0));

    //vec2 im = iMouse.xy / iResolution.xy;  
    //envOrientation = sphericalMatrix(im * PI * 2.);


    vec2 p = (-iResolution.xy + 2.* fragCoord) / iResolution.y;
    
    //return vec4(-face(p.xy), 0, 1);
    
    //p /= 2.;

    
    // jitter for antialiasing
    p += 2. * (seed - .5) / iResolution.xy;

    vec3 col = vec3(0);

    float focalLength = 3.;
    vec3 camPos = vec3(0, 0, 1.5) * focalLength;
    vec3 camTar = vec3(0, 0, 0);
    
    camPos.xy += rndcircle(seed) * .05;
    
    seed = hash22(seed);
    
    vec3 ww = normalize(camTar - camPos);
    vec3 uu = normalize(cross(vec3(0,1,0),ww));
    vec3 vv = normalize(cross(ww,uu));
    mat3 camMat = mat3(-uu, vv, ww);
    
    vec3 rayDir = normalize(camMat * vec3(p.xy, focalLength));
    vec3 origin = camPos;
    
	//origin = vec3(0,0,9.5);
   	//rayDir = normalize(vec3(p * .168, -1.));    

    Hit hit = march(origin, rayDir, 6. * focalLength, 1.);

    float firstHitLen = hit.len;

    vec3 nor, ref, raf;
    Material material;
    vec3 accum = vec3(1);
    vec3 bgCol = BGCOL;

    int BOUNCE = hit.model.id == 1 ? 10 : 5;
    
    float wavelength = seed.y;
    float extinctionDist = 0.;
    
    float ior, offset;
    
    bool spec = hit.model.id == 1;

    for (int bounce = 0; bounce < BOUNCE; bounce++) {
   
        if (bounce > 0) {
           seed = hash22(seed);
           hit = march(origin, rayDir, 6., 1.);
        }
        
        if (invert < 0.) {
            extinctionDist += hit.len;
        }

        if (hit.model.id == 0) {
            break;
        }

        nor = calcNormal(hit.pos);
        
        material = shadeModel(hit.model, nor);

        // update the colorMultiplier
       	accum *= material.albedo;


        if (hit.model.id == 1) {
            
            ref = reflect(rayDir, nor);
            
            // shade
            col += light(hit.pos, ref) * .5;
            col += pow(max(1. - abs(dot(rayDir, nor)), 0.), 5.) * .1;
            col *= vec3(.85,.85,.98);

            // refract
            ior = mix(1.2, 1.8, wavelength);
            ior = invert < 0. ? ior : 1. / ior;
            raf = refract(rayDir, nor, ior);
            bool tif = raf == vec3(0); // total internal reflection
            rayDir = tif ? ref : raf;
            invert *= -1.; // not correct but gives more interesting results
            
        } else {
            
            // Calculate diffuse ray direction
            seed = hash22(seed);
            vec3 diffuseRayDir = getSampleBiased(nor, 1., seed);

/*            
            // calculate direct lighting
            vec3 lightDir = (sunPos - hit.pos);
            vec3 lightSampleDir = getConeSample(lightDir, 1e-3, seed);
            float diffuse = dot(nor, lightSampleDir);
            vec3 shadowOrigin = hit.pos + nor * (.0002 / abs(dot(lightSampleDir, nor)));
            if (diffuse > 0.) {
                Hit sh = march(shadowOrigin, lightSampleDir, 3., 1.);
                if (sh.model.id == 0) {
                    col += accum * diffuse * (1./dot(lightDir, lightDir)) * .1;
                }
            }
  */          
            
            

            rayDir = diffuseRayDir;            
        }
        
        offset = .01 / abs(dot(rayDir, nor));
        origin = hit.pos + offset * rayDir;
    }    
    
    //col += env(hit.pos, rayDir) * accum;
    
    if (! spec) {
        col *= 2.;
    }

    vec3 fogcol = vec3(.001);
   // fogcol = vec3(1,0,0);
    //bgCol * .15
    //col = mix(col, fogcol, saturate(1.0 - exp2(-.0015 * pow(firstHitLen - length(camPos*.5), 3.))));
    col = mix(col, fogcol, saturate(1.0 - exp2(-.0006 * pow(firstHitLen - length(camPos*.666), 5.))));


        col *= spectrum(-wavelength+.25);
    

    return vec4(col, 1);
}


void main() {
    gl_FragColor = draw(gl_FragCoord.xy);
}

