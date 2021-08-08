/* File generated with Shader Minifier 1.1.6
 * http://www.ctrl-alt-test.fr
 */
#ifndef FRAG_DRAW_H_
# define FRAG_DRAW_H_
# define VAR_IFRAME "m"
# define VAR_IRESOLUTION "v"

const char *draw_frag =
 "#version 430\n"
 "layout(location=0)uniform vec4 v;"
 "layout(location=1)uniform int m;\n"
 "#define PI 3.1415926\n"
 "vec3 s(in float v,in vec3 m,in vec3 y,in vec3 i,in vec3 x)"
 "{"
   "return m+y*cos(6.28318*(i*v+x));"
 "}"
 "vec3 s(float v)"
 "{"
   "return s(v,vec3(.5,.5,.5),vec3(.5,.5,.5),vec3(1.,1.,1.),vec3(0.,.33,.67));"
 "}"
 "vec2 n(vec2 v)"
 "{"
   "v+=1.61803;"
   "vec3 f=fract(vec3(v.xyx)*vec3(.1031,.103,.0973));"
   "f+=dot(f,f.yzx+33.33);"
   "return fract((f.xx+f.yz)*f.zy);"
 "}"
 "const uint d=1103515245U;"
 "vec3 t(vec3 v)"
 "{"
   "uvec3 f=uvec3(v);"
   "f=(f>>8U^f.yzx)*d;"
   "f=(f>>8U^f.yzx)*d;"
   "f=(f>>8U^f.yzx)*d;"
   "return vec3(f)*(1./float(-1U));"
 "}"
 "float x(vec3 v)"
 "{"
   "return t(v).x;"
 "}"
 "vec2 e(vec2 v)"
 "{"
   "float f=v.x*2.*PI,i=sqrt(v.y);"
   "return vec2(i*cos(f),i*sin(f));"
 "}"
 "vec3 f(vec2 v)"
 "{"
   "return vec3(.2);"
 "}"
 "bool l;\n"
 "#define saturate(x)clamp(x,0.,1.)\n"
 "float e(float v,float y,float x)"
 "{"
   "float f=clamp(.5+.5*((v-y)/x),0.,1.);"
   "return(1.-f)*v+f*y-f*(1.-f)*x;"
 "}"
 "float f(float v,float m,float f)"
 "{"
   "return-e(-v,-m,f);"
 "}"
 "float n(float v,float f,float y)"
 "{"
   "vec2 x=max(vec2(y-v,y-f),vec2(0));"
   "return max(y,min(v,f))-length(x);"
 "}"
 "float s(float v,float f,float y)"
 "{"
   "vec2 x=max(vec2(y+v,y+f),vec2(0));"
   "return min(-y,max(v,f))+length(x);"
 "}"
 "float t(float v,float f,float x)"
 "{"
   "return min(e(v,f,x),n(v,f,x));"
 "}"
 "float x(float v,float m,float x)"
 "{"
   "return max(f(v,m,x),s(v,m,x));"
 "}"
 "void e(inout vec2 v,float x)"
 "{"
   "v=cos(x)*v+sin(x)*vec2(v.y,-v.x);"
 "}"
 "void r(inout vec2 v)"
 "{"
   "v=(v+vec2(v.y,-v.x))*sqrt(.5);"
 "}"
 "vec3 f(vec3 v,float f)"
 "{"
   "return e(v.yz,f),v;"
 "}"
 "vec3 n(vec3 v,float f)"
 "{"
   "return e(v.xz,f),v;"
 "}"
 "vec3 r(vec3 v,float f)"
 "{"
   "return e(v.xy,f),v;"
 "}"
 "float h(vec3 v)"
 "{"
   "return min(min(v.x,v.y),v.z);"
 "}"
 "float p(vec3 v)"
 "{"
   "return max(max(v.x,v.y),v.z);"
 "}"
 "float w(vec2 v)"
 "{"
   "return max(v.x,v.y);"
 "}"
 "float h(vec3 v,vec3 x)"
 "{"
   "vec3 f=abs(v)-x;"
   "return length(max(f,vec3(0)))+p(min(f,vec3(0)));"
 "}"
 "float p(vec2 v,vec2 x)"
 "{"
   "vec2 f=abs(v)-x;"
   "return length(max(f,vec2(0)))+w(min(f,vec2(0)));"
 "}struct Material{vec3 albedo;float specular;float roughness;};struct Model{float d;vec3 uvw;vec3 albedo;int id;};"
 "Material s(Model v,inout vec3 x)"
 "{"
   "int f=v.id;"
   "vec3 m=v.uvw;"
   "return Material(v.albedo,0.,0.);"
 "}"
 "float i;"
 "mat3 t(vec3 v,vec3 f)"
 "{"
   "vec3 x=normalize(v),i=normalize(cross(x,f)),y=normalize(cross(i,x));"
   "return mat3(i,x,y);"
 "}"
 "vec3 h(vec3 v,vec3 f,vec3 x)"
 "{"
   "return v*t(f,x);"
 "}"
 "float w(vec3 v,vec3 f)"
 "{"
   "float i=h(f,v);"
   "i=max(i,-abs(f.x));"
   "i=max(i,-(i+h(v)*.333));"
   "return i;"
 "}"
 "float p(vec3 v,vec3 f,vec3 m)"
 "{"
   "m.y=max(m.y,.5*v.y/f.y);"
   "m.y-=v.y*.5;"
   "v.y*=.5;"
   "vec3 n=m;"
   "float i=1e+12;"
   "for(int y=0;y<int(f.x);y++)"
     "{"
       "for(int r=0;r<int(f.y);r++)"
         "{"
           "for(int d=0;d<int(f.z);d++)"
             "{"
               "m=n;"
               "vec3 l=vec3(y,r,d);"
               "m-=((l+.5)/f-.5)*v*2.;"
               "vec3 s=v/f,c=t(l+11.);"
               "m-=(c*2.-1.)*s*.5;"
               "float z=x(l*10.+27.);"
               "s*=mix(.6,1.5,z);"
               "s.xz*=mix(1.8,.45,pow(float(r)/(f.y-1.),.5));"
               "float M=h(m,s);"
               "M=max(M,-abs(m.x));"
               "if(c.z>.5&&l.y>0.)"
                 "M=max(M,-abs(m.y-(z*2.-1.)*s.y*.5));"
               "i=min(i,M);"
             "}"
         "}"
     "}"
   "i=max(i,-(i+h(v/f)*.4));"
   "return i;"
 "}"
 "float r(vec3 v,vec3 f,vec3 m)"
 "{"
   "v*=.9;"
   "m.y-=v.y*.5;"
   "v.y*=.5;"
   "vec3 n=m;"
   "float i=1e+12;"
   "for(int y=0;y<int(f.x);y++)"
     "{"
       "for(int r=0;r<int(f.y);r++)"
         "{"
           "for(int d=0;d<int(f.z);d++)"
             "{"
               "m=n;"
               "vec3 l=vec3(y,r,d);"
               "m-=((l+.5)/f-.5)*v*2.;"
               "vec3 s=v/f;"
               "float M=x(l+15.);"
               "s*=mix(1.1,1.75,M);"
               "float c=h(m,s)+.01;"
               "if(l==vec3(0))"
                 "c=max(c,-abs(m.x));"
               "c=max(c,-i);"
               "i=min(i,c);"
             "}"
         "}"
     "}"
   "i=max(i,-(i+h(v/f)*.5));"
   "return i;"
 "}"
 "float c(vec3 v)"
 "{"
   "float f=1e+12,x=.2;"
   "f=p(vec3(.35,1.6,.35),vec3(2,3,2),h(v-vec3(.8,0,-.8),vec3(.2,1,-.5),vec3(1,0,1)));"
   "f=e(f,w(vec3(.13),h(v-vec3(1.8,-.15,-.3),vec3(0,1,0),vec3(1,0,-.25))),x);"
   "f=e(f,r(vec3(.3,.35,.3),vec3(2,1,2),h(v-vec3(-.3,0,.5),vec3(0.,1,.2),vec3(0.,0,1))-vec3(0,-.2,0)),x);"
   "return f;"
 "}"
 "Model P(vec3 v)"
 "{"
   "vec2 f=vec2(.43,.43);"
   "e(v.yz,(.5-f.y)*PI);"
   "e(v.xz,(.5-f.x)*PI*2.5);"
   "v.y+=.6;"
   "v.xz-=vec2(-1,1)*.4;"
   "float y=7.,m=c(v),r=v.y+.25;"
   "r=e(r,length(v-vec3(.6,-2.5,-.7))-2.5,.6);"
   "r=e(r,length(v-vec3(-.3,-.5,.5))-.5,.4);"
   "float x=pow(m+.333,.5)*1.5;"
   "x=m;"
   "r+=cos(max(x,0.)*y*PI*2.)*.015;"
   "Model d=Model(r,v,vec3(3),3),n=Model(m*i,v,vec3(1),1);"
   "if(n.d<d.d)"
     "d=n;"
   "return d;"
 "}"
 "const float y=1.73205;"
 "vec3 M(in vec3 v)"
 "{"
   "vec3 f=vec3(0.);"
   "for(int i=0;i<4;i++)"
     "{"
       "vec3 x=.5773*(2.*vec3(i+3>>1&1,i>>1&1,i&1)-1.);"
       "f+=x*P(v+.001*x).d;"
     "}"
   "return normalize(f);"
 "}"
 "vec3 a=normalize(vec3(5,2,2))*2.;struct Hit{Model model;vec3 pos;float len;};"
 "Hit M(vec3 v,vec3 f,float x,float y)"
 "{"
   "vec3 i;"
   "float m=0.,r=0.;"
   "Model n;"
   "float d=0.;"
   "for(float M=0.;M<300.;M++)"
     "{"
       "m+=r;"
       "i=v+m*f;"
       "n=P(i);"
       "r=n.d;"
       "d+=1.;"
       "if(abs(n.d)/m<.0002)"
         "{"
           "break;"
         "}"
       "if(m>=x)"
         "{"
           "m=x;"
           "n.id=0;"
           "break;"
         "}"
     "}"
   "return Hit(n,i,m);"
 "}"
 "vec3 u(vec3 v)"
 "{"
   "vec3 f=cross(vec3(-1,-1,.5),v);"
   "return f;"
 "}"
 "vec3 M(vec3 v,float f,vec2 x)"
 "{"
   "v=normalize(v);"
   "vec3 y=normalize(u(v)),i=normalize(cross(v,y));"
   "vec2 m=x;"
   "m.x=m.x*2.*PI;"
   "m.y=pow(m.y,1./(f+1.));"
   "float d=sqrt(1.-m.y*m.y);"
   "return cos(m.x)*d*y+sin(m.x)*d*i+m.y*v;"
 "}"
 "vec3 P(vec3 v,float f,vec2 x)"
 "{"
   "v=normalize(v);"
   "vec3 y=normalize(u(v)),i=normalize(cross(v,y));"
   "vec2 m=x;"
   "m.x=m.x*2.*PI;"
   "m.y=1.-m.y*f;"
   "float d=sqrt(1.-m.y*m.y);"
   "return cos(m.x)*d*y+sin(m.x)*d*i+m.y*v;"
 "}"
 "vec3 o=vec3(.9,.83,1);"
 "float M(vec3 v,vec3 f,vec3 m,vec3 x,vec3 y,out vec2 i)"
 "{"
   "float d=dot(x,m-v)/dot(f,x);"
   "vec3 r=v+d*f,c=cross(x,y),l=cross(x,c);"
   "r-=m;"
   "i=vec2(dot(c,r),dot(l,r));"
   "return max(sign(d),0.);"
 "}"
 "mat3 H(vec2 v)"
 "{"
   "float f=v.x,x=v.y,y=cos(f),i=cos(x),m=sin(f),d=sin(x);"
   "return mat3(i,-d*-m,-d*y,0,y,m,d,i*-m,i*y);"
 "}"
 "mat3 b;"
 "vec3 H(vec3 v,vec3 f)"
 "{"
   "v=-v;"
   "f=-f;"
   "v*=b;"
   "f*=b;"
   "vec2 x;"
   "vec3 i=vec3(-6);"
   "float y=M(v,f,i,normalize(i),normalize(vec3(-1,1,0)),x),m=smoothstep(.75,0.,p(x,vec2(.5,2))-1.);"
   "m*=smoothstep(6.,0.,length(x));"
   "return vec3(m)*y*2.;"
 "}"
 "void main()"
 "{"
   "vec2 f=n(gl_FragCoord.xy+float(m)*y);"
   "i=1.;"
   "b=H((vec2(81.5,119)/vec2(187)*2.-1.)*2.);"
   "vec2 r=(-v.xy+2.*gl_FragCoord.xy)/v.y;"
   "r+=2.*(f-.5)/v.xy;"
   "vec3 x=vec3(0);"
   "float d=3.;"
   "vec3 c=vec3(0,0,1.5)*d,l=vec3(0,0,0.);"
   "f=n(f);"
   "vec3 z=normalize(l-c),a=normalize(cross(vec3(0,1,0),z)),p=normalize(cross(z,a));"
   "mat3 u=mat3(-a,p,z);"
   "vec3 P=normalize(u*vec3(r.xy,d)),g=c;"
   "Hit t=M(g,P,6.*d,1.);"
   "float e=t.len;"
   "vec3 I,U,k;"
   "Material w;"
   "vec3 h=vec3(1),q=o;"
   "int F=t.model.id==1?10:5;"
   "float C=f.y,Z=0.,Y,X;"
   "bool W=t.model.id==1;"
   "for(int V=0;V<F;V++)"
     "{"
       "if(V>0)"
         "f=n(f),t=M(g,P,6.,1.);"
       "if(i<0.)"
         "Z+=t.len;"
       "if(t.model.id==0)"
         "{"
           "break;"
         "}"
       "I=M(t.pos);"
       "w=s(t.model,I);"
       "h*=w.albedo;"
       "if(t.model.id==1)"
         "{"
           "U=reflect(P,I);"
           "x+=H(t.pos,U)*.5;"
           "x+=pow(max(1.-abs(dot(P,I)),0.),5.)*.1;"
           "x*=vec3(.85,.85,.98);"
           "Y=mix(1.2,1.8,C);"
           "Y=i<0.?Y:1./Y;"
           "k=refract(P,I,Y);"
           "bool T=k==vec3(0);"
           "P=T?U:k;"
           "i*=-1.;"
         "}"
       "else"
         "{"
           "f=n(f);"
           "vec3 T=M(I,1.,f);"
           "P=T;"
         "}"
       "X=.01/abs(dot(P,I));"
       "g=t.pos+X*P;"
     "}"
   "if(!W)"
     "x*=3.;"
   "x*=s(-C+.25);"
   "vec3 T=vec3(.001);"
   "x=mix(x,T,saturate(1.-exp2(-.0015*pow(e-length(c*.5),3.))));"
   "gl_FragColor=vec4(x,1);"
 "}";

#endif // FRAG_DRAW_H_
