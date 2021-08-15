/* File generated with Shader Minifier 1.1.6
 * http://www.ctrl-alt-test.fr
 */
#ifndef FRAG_PRESENT_H_
# define FRAG_PRESENT_H_
# define VAR_ACCUMULATORTEX "x"
# define VAR_IRESOLUTION "v"

const char *present_frag =
 "#version 430\n"
 "layout(location=0)uniform vec4 v;"
 "layout(binding=0)uniform sampler2D x;\n"
 "#define sat(x)clamp(x,0.,1.)\n"
 "vec3 t(float v)"
 "{"
   "float x,r,s;"
   "if(v<=66.)"
     "{"
       "x=1.;"
       "r=sat((99.4708*log(v)-161.12)/255.);"
       "if(v<19.)"
         "s=0.;"
       "else"
         " s=sat((138.518*log(v-10.)-305.045)/255.);"
     "}"
   "else"
     " x=sat(1.29294*pow(v-60.,-.133205)),r=sat(1.12989*pow(v-60.,-.0755148)),s=1.;"
   "return vec3(x,r,s);"
 "}"
 "vec3 s(vec3 v)"
 "{"
   "const float x=2.51,s=.03,r=2.43,i=.59,n=.14;"
   "return clamp(v*(x*v+s)/(v*(r*v+i)+n),0.,1.);"
 "}"
 "float p(vec3 v)"
 "{"
   "return dot(v,vec3(.2126,.7152,.0722));"
 "}"
 "void main()"
 "{"
   "vec4 v=texelFetch(x,ivec2(gl_FragCoord.xy),0);"
   "vec3 r=v.xyz/v.w;"
   "float n=p(r);"
   "r=mix(vec3(n),r,1.25);"
   "r*=1./t(100.);"
   "vec3 i=vec3(1.333),m=vec3(.0015,0.,.005)*1.25,g=vec3(0.,0.,0.),y=vec3(.0666);"
   "r=pow(max(vec3(0.),r*(1.+i-m)+m+g),max(vec3(0.),1.-y));"
   "r=s(r);"
   "r=pow(r,vec3(1./2.2));"
   "gl_FragColor=vec4(r,1);"
 "}";

#endif // FRAG_PRESENT_H_
