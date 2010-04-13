#ifndef __READIM7_H
#define __READIM7_H

//***************************************************This is where ReadIMX.h start**********************************************

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef enum
{
        IMREAD_ERR_NO = 0,      // no error
        IMREAD_ERR_FILEOPEN,    // cannot open file
        IMREAD_ERR_HEADER,      // error while reading the header
        IMREAD_ERR_FORMAT,      // file format not read by this DLL
        IMREAD_ERR_DATA,        // data reading error
        IMREAD_ERR_MEMORY,      // out of memory
} ImReadError_t;


typedef unsigned char      Byte;
typedef unsigned short int Word;

struct BufferScaleType_
{
        float   factor;
        float   offset;
        char    description[16];
        char    unit[16];
}; 
typedef	struct BufferScaleType_ BufferScaleType;

enum BufferFormat_t
{
        BUFFER_FORMAT__NOTUSED          = -1,   // not used any longer
        BUFFER_FORMAT_MEMPACKWORD       = -2,   // memory packed Word buffer (= byte buffer)
        BUFFER_FORMAT_FLOAT             = -3,   // float image
        BUFFER_FORMAT_WORD              = -4,   // word image
        BUFFER_FORMAT_DOUBLE            = -5,   // double image
        BUFFER_FORMAT_FLOAT_VALID       = -6,   // float image with flags for valid pixels
        BUFFER_FORMAT_IMAGE             = 0,    //image with unspecified data format (word, float or double)
        BUFFER_FORMAT_VECTOR_2D_EXTENDED,       //PIV vector field with header and 4*2D field
        BUFFER_FORMAT_VECTOR_2D,                //simple 2D vector field
        BUFFER_FORMAT_VECTOR_2D_EXTENDED_PEAK,  //same as 1 + peak ratio
        BUFFER_FORMAT_VECTOR_3D,                //simple 3D vector field
        BUFFER_FORMAT_VECTOR_3D_EXTENDED_PEAK,  //PIV vector field with header and 4*3D field + peak ratio
        BUFFER_FORMAT_COLOR             = -10,  // base of color types
        BUFFER_FORMAT_RGB_MATRIX        = -10,  // RGB matrix from color camera (Imager3)
        BUFFER_FORMAT_RGB_32            = -11   // each pixel has 32bit RGB color info
};

struct BufferType_
{
   int         isFloat;
   int         nx,ny,nz,nf;
   int         totalLines;
   int         vectorGrid;     // 0 for images
   int         image_sub_type; // BufferFormat_t
   union {
      float*   floatArray;
      Word*    wordArray;
   };
   BufferScaleType scaleX; // x-scale
   BufferScaleType scaleY; // y-scale
   BufferScaleType scaleI; // intensity scale
};

typedef struct BufferType_ BufferType;

struct AttributeList_
{
   char*          name;
   char*          value;
   struct AttributeList_ * next;
};

typedef struct AttributeList_ AttributeList;

enum    Image_t // type in header of IMX files
{       IMAGE_IMG = 18,    // uncompressed WORD image
	IMAGE_IMX,         // compressed WORD image
	IMAGE_FLOAT,       // floating point uncompressed
	IMAGE_SPARSE_WORD, // sparse buffer with word scalars
	IMAGE_SPARSE_FLOAT,// sparse buffer with float scalars
	IMAGE_PACKED_WORD, // memory packed word buffer with one byte per pixel
};

Byte* Buffer_GetRowAddrAndSize(BufferType * myBuffer, int theRow, unsigned long * theRowLength );
int  CreateBuffer( BufferType * myBuffer, int theNX, int theNY, int theNZ, int theNF, int isFloat, int vectorGrid, enum BufferFormat_t imageSubType );
void SetBufferScale( BufferScaleType* theScale, float theFactor, float theOffset, const char* theDesc, const char* theUnit );

//! Destroy the data structure creacted by ReadIMX().
void DestroyBuffer( BufferType* myBuffer );
void DestroyAttributeList( AttributeList* myList );

void WriteAttribute_END( FILE *theFile );

int ReadImgExtHeader( FILE* theFile, AttributeList** myList );

ImReadError_t SCPackOldIMX_Read( FILE* theFile, BufferType* myBuffer );


//**********************************This is where ReadIMX.h ends*********************************************

//**********************************This is where ReadIM7.h starts*******************************************


enum ImageExtraFlag_t	// bit flags !
	{	IMAGE_EXTRA_FRAMETABLE	= 1,	// not used yet, TL 29.01.03
		IMAGE_EXTRA_2		= 2,
		IMAGE_EXTRA_3		= 4,
	};

typedef struct		// image file header (256 Bytes) for DaVis 7
{
	short int version;       // 0: file version, increased by one at every header change
	short int pack_type;     // 2: IM7PackType_t
	short int buffer_format; // 4: BufferFormat_t
	short int isSparse;      // 6: no (0), yes (1)
	int	  sizeX;	 // 8
	int	  sizeY;	 // 12
	int   sizeZ;	 // 16
	int	  sizeF;	 // 20
	short int scalarN;	 // 24: number of scalar components
	short int vector_grid; 	 // 26: 1-n = vector data: grid size
	short int extraFlags;	 // 28: ImageExtraFlag_t
	// 30:
	char	  reserved[256-30];
} Image_Header_7;




// Returns error code ImReadError_t, can read IM7, VC7 and IMX, IMG, VEC
int ReadIM7 ( const char* theFileName, BufferType* myBuffer, AttributeList** myList );

//**********************************This is where ReadIM7.h ends**********************************************


//**********************************This is where mperl starts***********************************************


#endif //__READIM7_H
