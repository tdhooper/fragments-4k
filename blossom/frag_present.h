/* File generated with Shader Minifier 1.1.6
 * http://www.ctrl-alt-test.fr
 */
#ifndef FRAG_PRESENT_H_
# define FRAG_PRESENT_H_
# define VAR_ACCUMULATORTEX "v"
# define VAR_IRESOLUTION "c"

const char *present_frag =
 "#version 430\n"
 "layout(location=0)uniform vec4 c;"
 "layout(binding=0)uniform sampler2D v;"
 "vec3 t(vec3 v)"
 "{"
   "const float c=2.51,r=.03,m=2.43,n=.59,i=.14;"
   "return clamp(v*(c*v+r)/(v*(m*v+n)+i),0.,1.);"
 "}"
 "float x(vec3 v)"
 "{"
   "return dot(v,vec3(.2126,.7152,.0722));"
 "}"
 "vec3 w(vec3 v)"
 "{"
   "v/=2.;"
   "v*=16.;"
   "vec3 c=max(vec3(0),v-.004);"
   "return c*(6.2*c+.5)/(c*(6.2*c+1.7)+.06);"
 "}"
 "void main()"
 "{"
   "vec4 c=texelFetch(v,ivec2(gl_FragCoord.xy),0);"
   "vec3 m=c.xyz/c.w;"
   "float r=x(m);"
   "m=mix(vec3(r),m,1.25);"
   "vec3 n=vec3(4.),i=vec3(.002,0.,.005),g=vec3(0.,0.,0.),y=vec3(-.35);"
   "m=pow(max(vec3(0.),m*(1.+n-i)+i+g),max(vec3(0.),1.-y));"
   "m=pow(m,vec3(1./2.2));"
   "m=t(m);"
   "gl_FragColor=vec4(m,1);"
 "}";

#endif // FRAG_PRESENT_H_
