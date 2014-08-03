#ifndef GL_HELPER
#define GL_HELPER

#include "glee.h"
#include <gl/glext.h>	
#include <gl/glut.h>

#include "common_defs.h"

#include <Cg/cgGL.h>

#include "image.h"
#include "mtime.h"

extern void checkOpenGL ();
extern void drawText2D ( int x, int y, char* msg);
extern void drawText3D ( float x, float y, float z, char* msg);
extern void drawGrid ();
extern void measureFPS ();

extern mint::Time	tm_last;
extern int			tm_cnt;
extern float		tm_fps;

extern CGprogram	cgVP;
extern CGprogram	cgFP;
extern CGprofile	vert_profile;
extern CGprofile	frag_profile;

extern void disableShadows ();
extern void checkFrameBuffers ();
extern int createShader ( int n, std::string vname, std::string vfunc, std::string fname, std::string ffunc);

extern GLuint glSphere;
extern float  glRadius;
extern void setSphereRadius ( float f );
extern void drawSphere ();

#endif