class Image;

#ifndef IMAGE_H
#define IMAGE_H

#include "common_defs.h"
#include <stdio.h>

#ifdef _MSC_VER
#include <windows.h>
#endif

#define IRED	0
#define IGREEN	1
#define IBLUE	2
#define IALPHA	3

class Pixel {
public:
	Pixel ()	{ r=0;g=0;b=0;a=1;}
	Pixel ( double r_, double g_, double b_ )	{r = r_; g = g_; b = b_; a = 1.0; }
	Pixel ( double r_, double g_, double b_, double a_ )	{r = r_; g = g_; b = b_; a = a_; }
	double		r;
	double		g;
	double		b;
	double		a;
};

class Image {
public:
	Image ();
	~Image ();

	Image (int width_, int height_);
	Image (int width_, int height_, int channels_);
	Image (int width_, int height_, int channels_, 
		int bits_);

	Image (const char* filename);

	Image (const Image& image);
	Image&		operator= (const Image& image);

	int		getWidth ()    { return width;    }
	int		getHeight ()   { return height;   }
	int		getChannels () { return channels; }
	int		getBits ()     { return bits;     }
	unsigned char*	getPixels ()   { return pixels;   }

	void	create ( int width_, int height_, int channels_ );

	void	refresh ();

	void	draw ();
	void	draw ( float x, float y );

	unsigned char*	getPixelData ()		{ return pixels; }
	void		setPixels ( unsigned char *newPixels );

	bool		good ();
	bool		bad ();

	void		clear ();
	void		clear ( Pixel pixel );

	double		getPixel  (int x, int y, int channel);
	double		getPixel_ (int x, int y, int channel);
	Pixel		getPixel  (int x, int y);
	Pixel		getPixel_ (int x, int y);
	Pixel&		getPixel  (int x, int y, Pixel& pixel);
	Pixel&		getPixel_ (int x, int y, Pixel& pixel);

	void		setPixel  (int x, int y, int channel, double value);
	void		setPixel_ (int x, int y, int channel, double value);
	void		setPixel  (int x, int y, Pixel pixel);
	void		setPixel_ (int x, int y, Pixel pixel);
	void		setPixel4 ( int x, int y, Pixel pixel );

	void		setAlpha (int x, int y, double value);

#ifndef DISABLE_OPENGL
	void		glReadPixelsWrapper ();
	void		glDrawPixelsWrapper ();
	void		glTexImage2DWrapper ();
	void		glTexImageCubeWrapper ( int i );
	void		glTexSubImage2DWrapper ( int x, int y);
#endif

	int		getID ()		{ return imgID; }
	void    generateID();

	int		read (const char* filename);
	int		read (const char* filename, const char* alphaname );
	int		write (const char* filename);

	int		readPaletteBMP ( FILE* fp, RGBQUAD*& palette, int bit_count );

	int		readBMP (const char* filename);
	int		readBMP (FILE* file, FILE* file_a, bool bBaseImg );
	int		writeBMP (const char* filename);
	int		writeBMP (FILE* file);

	int		readPNM (const char* filename);
	int		readPNM (FILE* file);
	int		writePNM (const char* filename);
	int		writePNM (FILE* file);

private:

	int		index(int x, int y, int c);

	int		width;
	int		height;
	int		channels;	
	int		bits;		
	int		maxValue;	

	unsigned char*	pixels;		

	bool		owns;		

	unsigned int		imgID;

};


#endif // IMAGE_H
