#include "common_defs.h"
#include "gl_helper.h"

#include <math.h>

float light_proj[16];
float light_x, light_y, light_z;
float light_tox, light_toy, light_toz;
float light_mfov;

// ×ÖÌå
void *font = GLUT_BITMAP_8_BY_13;
void *fonts[] = {GLUT_BITMAP_9_BY_15,
	GLUT_BITMAP_TIMES_ROMAN_10,
	GLUT_BITMAP_TIMES_ROMAN_24};

// ¼ÆÊ±
mint::Time	tm_last;
int			tm_cnt;
float		tm_fps;

GLuint glSphere = 65535;
float  glRadius = 0.0;

CGcontext cgContext = 0;
CGprofile vert_profile = CG_PROFILE_UNKNOWN;
CGprofile frag_profile = CG_PROFILE_UNKNOWN;
CGprogram cgVP = 0;
CGprogram cgFP = 0;

void cgErrorCallback ()
{
	CGerror cerror = cgGetError();
	if ( cerror ) {
		const char *listing = cgGetLastListing ( cgContext );		
		printf ( "CG: %s\n%s\n", cgGetErrorString( cerror ), listing );
		exit (-1);
	}
}


int createShader ( int n, std::string vname, std::string vfunc, std::string fname, std::string ffunc)
{
	char vnbuf[200];
	char vfbuf[200];
	char fnbuf[200];
	char ffbuf[200];
	strcpy ( vnbuf, vname.c_str() );
	strcpy ( vfbuf, vfunc.c_str() );
	strcpy ( fnbuf, fname.c_str() );
	strcpy ( ffbuf, ffunc.c_str() );

	if ( cgContext == 0 ) {
		cgSetErrorCallback( cgErrorCallback );
		cgContext = cgCreateContext();
	}

	// Select profiles
	vert_profile = cgGLGetLatestProfile ( CG_GL_VERTEX );
	cgGLSetOptimalOptions( vert_profile );

	frag_profile = cgGLGetLatestProfile ( CG_GL_FRAGMENT );
	cgGLSetOptimalOptions( frag_profile );

	printf ( "Vertex profile:   %s\n", cgGetProfileString(vert_profile) );
	printf ( "Fragment profile: %s\n", cgGetProfileString(frag_profile) );
	printf ( " (See http.developer.nvidia.com/Cg/index_profiles.html)\n");

	printf ( "Loading VP:       %s\n", vnbuf );
	cgVP = cgCreateProgramFromFile( cgContext, CG_SOURCE, vnbuf, vert_profile, vfbuf, NULL );
	cgGLLoadProgram( cgVP );

	printf ( "Loading FP:       %s\n", fnbuf );
	cgFP = cgCreateProgramFromFile( cgContext, CG_SOURCE, fnbuf, frag_profile, ffbuf, NULL );	
	cgGLLoadProgram( cgFP );

	cgGLSetManageTextureParameters ( cgContext, CG_FALSE ); 
	cgGLBindProgram ( cgVP );
	cgGLBindProgram ( cgFP );	

	return 0;
}

void setSphereRadius ( float r )
{
	if ( glRadius == r ) return;
	glRadius = r;

	int udiv = 6;
	int vdiv = 6;

	if ( glSphere != 65535 ) glDeleteLists ( glSphere, 1 );
	glSphere = glGenLists ( 1 );
	float x, y, z, x1, y1, z1;	

	float du = 180.0 / udiv;
	float dv = 360.0 / vdiv;

	glNewList ( glSphere, GL_COMPILE );
	glBegin ( GL_TRIANGLE_STRIP );
	for ( float tilt=-90; tilt <= 90.0; tilt += du) {
		for ( float ang=0; ang <= 360; ang += dv) {
			x = sin ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			y = cos ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			z = sin ( tilt*DEGtoRAD ) ;
			x1 = sin ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			y1 = cos ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			z1 = sin ( (tilt+du)*DEGtoRAD );
			glNormal3f ( x, y, z );		glVertex3f ( x*r, y*r, z*r );		
			glNormal3f ( x1, y1, z1 );	glVertex3f ( x1*r, y1*r, z1*r );
		}
	}
	glEnd ();
	glEndList ();
}

void drawSphere ()
{
	if ( glRadius == 0.0 ) 
		setSphereRadius ( 2 );		
	glCallList ( glSphere );
}

void checkOpenGL ()
{
	GLenum errCode = glGetError();
	if (errCode != GL_NO_ERROR) {
		const GLubyte* errString = gluErrorString(errCode);
		fprintf( stderr, "OpenGL error: %s\n", errString );
	}
}

void drawText2D ( int x, int y, char* msg)
{
	int len, i;
	glRasterPos2f(x, y);
	len = (int) strlen(msg);
	for (i = 0; i < len; i++) 
		glutBitmapCharacter(font, msg[i]);  
}

void drawText3D ( float x, float y, float z, char* msg)
{
	int len, i;
	glRasterPos3f(x, y, z);
	len = (int) strlen(msg);
	for (i = 0; i < len; i++) 
		glutBitmapCharacter(font, msg[i]);  
}

void drawGrid ()
{
	glColor3f ( 0.3, 0.3, 0.3 );
	glBegin ( GL_LINES );
	for (float x=-40; x<=40.0; x+=10.0 ) {
		glVertex3f ( x, -40.0, 0 );
		glVertex3f ( x,  40.0, 0 );
	}
	for (float y=-40; y<=40.0; y+=10.0 ) {
		glVertex3f ( -40.0, y, 0 );
		glVertex3f (  40.0, y, 0 );
	}
	glEnd ();
}

void measureFPS ()
{
	mint::Time tm_elaps;	
	if ( ++tm_cnt > 5 ) {		
		tm_elaps.SetSystemTime ( ACC_NSEC );			
		tm_elaps = tm_elaps - tm_last;					
		tm_fps = 5.0 * 1000.0 / tm_elaps.GetMSec ();	
		tm_cnt = 0;									
		tm_last.SetSystemTime ( ACC_NSEC );
	}
}

void checkFrameBuffers ()
{                                                            
	GLenum status;                                             
	status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);  
	switch(status) {                                          
	case GL_FRAMEBUFFER_COMPLETE_EXT: printf ( "FBO complete\n" ); break;                                                
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT: printf ( "FBO format unsupported\n"); break;                                                
	default:  printf ( "Unknown FBO error\n");
	}
}

void disableShadows ()
{
	glDisable ( GL_TEXTURE_2D );		

	glActiveTexture( GL_TEXTURE1 );
	glBindTexture ( GL_TEXTURE_2D, 0 );
	glDisable ( GL_TEXTURE_GEN_S );
	glDisable ( GL_TEXTURE_GEN_T );
	glDisable ( GL_TEXTURE_GEN_R );
	glDisable ( GL_TEXTURE_GEN_Q );	

	glActiveTexture( GL_TEXTURE2 );
	glBindTexture ( GL_TEXTURE_2D, 0 );		
	glDisable ( GL_TEXTURE_GEN_S );
	glDisable ( GL_TEXTURE_GEN_T );
	glDisable ( GL_TEXTURE_GEN_R );
	glDisable ( GL_TEXTURE_GEN_Q );	
}