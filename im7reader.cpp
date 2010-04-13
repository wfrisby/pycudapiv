#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "ReadIM7.h"

typedef struct		// image file header (256 Bytes)
	{
	 short int 	imagetype;      // 0:  (Image_t)
	 short int	xstart;		// 2:  start-pos left, not used
	 short int	ystart;		// 4:  start-pos top, not used
	 Byte		extended[4];	// 6:  reserved
	 short int	rows;		// 10: total number of rows, if -1 the real number is stored in longRows
	 short int	columns;	// 12: total number of columns, if -1 see longColumns
	 short int  image_sub_type; // 14: type of image (int):
					            //(included starting from sc_version 4.2 on)
					            //0 = normal image
                                //1 = PIV vector field with header and 4*2D field
                                //rows = 9 * y-size
                                //2 = simple 2D vector field (not used yet)
                                //rows = 2 * y-size
                                //3 ...
	 short int 	y_dim;		// 16: size of y-dimension (size of x-dimension is always = columns), not used
	 short int 	f_dim;		// 18: size of f-dimension (number of frames)
	 // for image_sub_type 1/2 only:
	 short int	vector_grid; 	// 20: 1-n = vector data: grid size
				      	//(included starting from sc_version 4.2 on)
	 char		ext[11];// 22: reserved

	 Byte		version;// 33:  e.g. 120 = 1.2	300+ = 3.xx  40-99 = 4.0 bis 9.9
#ifdef _WIN32
	 char       date[9];       // 34	
	 char       time[9];	   // 43
#else
//#message MIS-ALIGNED!!!
	 char       date[8];       // 34	
	 char       time[8];	   // 43
#endif
	 short int  xinit;         // 52:  x-scale values
	 float      xa;            // 54
	 float      xb;            // 58
	 char       xdim[11];      // 62
	 char       xunits[11];    // 73
	 short int  yinit;         // 84:  y-scale values
	 float      ya;            // 86
	 float      yb;            // 90
	 char       ydim[11];      // 94
	 char       yunits[11];    // 105
	 short int  iinit;         // 116: i-scale (intensity) values
	 float      ia;            // 118
	 float      ib;            // 122
	 char       idim[11];      // 126
	 char       iunits[11];    // 137
	 char       com1[40];      // 148: comment (2 lines)
	 char       com2[40];		// 188

	 int	    longRows;		// 228: (large) number of rows, TL 04.02.2000
	 int	    longColumns;	// 232: (large) number of columns, TL 04.02.2000
	 int	    longZDim;		// 236: (large) number of z-dimension, TL 02.05.00

	 char		reserved[12];	// 240: reserved
	 int		checksum;       // 252-255: not used
} image_header;

enum IM7PackType_t
{
	IM7_PACKTYPE__UNCOMPR= 0x1000,// autoselect uncompressed
	IM7_PACKTYPE__FAST,				// autoselect fastest packtype, 
	IM7_PACKTYPE__SIZE,				// autoselect packtype with smallest resulting file
	IM7_PACKTYPE_IMG			= 0,	// uncompressed, like IMG
	IM7_PACKTYPE_IMX,					// old version compression, like IMX
	IM7_PACKTYPE_ZLIB,				// Packen mit zlib
};


class IM7reader {
private:
size_t PAGESIZE; //= 0x800;		// Words !  size must not be too big!
long IMX_pagepos;// = 0;   // TL 16.12.1999

ImReadError_t SCPackOldIMX_Read( FILE* theFile, BufferType* myBuffer );
int CreateBuffer(BufferType * myBuffer, int theNX, int theNY, int theNZ, int theNF, int isFloat, int vectorGrid, enum BufferFormat_t imageSubType);
void SetBufferScale(BufferScaleType* theScale, float theFactor, float theOffset, const char* theDesc, const char* theUnit);
void CreatePinnedPool(void);
Byte * Buffer_GetRowAddrAndSize( BufferType* myBuffer, int theRow, unsigned long * theRowLength);
Word ReadNextPage(FILE* file, Word* page, signed char** bpageadr, Word* count);
void Scale_Read(const char*theData, BufferScaleType* theScale);
public:
int ReadIM7(const char* theFileName, BufferType* myBuffer, AttributeList** myList);
IM7reader(const char * fileName);
};

IM7reader::IM7reader(const char * fileName)
{
    PAGESIZE=0x800;
    IMX_pagepos=0;
    BufferType myBuffer;
    ReadIM7(fileName, &myBuffer, NULL);
}

// read data from the image file to the buffer, reset 2 variables
// called by SCBuffer:Read
Word IM7reader::ReadNextPage(FILE* file, Word* page, signed char** bpageadr, Word* count)
{
    IMX_pagepos = ftell(file);
    if (fread(page, 1, PAGESIZE, file)==0)
        return 0; // read error? (not TRUE if EOF)
    else
    {
        *count = 0;
        *bpageadr = (signed char*) page;
        return (Word)(ftell(file) - IMX_pagepos);
    }
}

Byte* IM7reader::Buffer_GetRowAddrAndSize( BufferType* myBuffer, int theRow, unsigned long * theRowLength )
{
	*theRowLength = myBuffer->nx * (myBuffer->isFloat?sizeof(float):sizeof(Word));
	if (myBuffer->isFloat)
	  return (Byte*)&(myBuffer->floatArray[myBuffer->nx*theRow]);
	return (Byte*)&(myBuffer->wordArray [myBuffer->nx*theRow]);
}

void IM7reader::SetBufferScale(BufferScaleType* theScale, float theFactor, float theOffset, const char* theDesc, const char* theUnit )
{
	if (theScale)
	{ 
		theScale->factor = theFactor;
		theScale->offset = theOffset;
		strncpy( theScale->description, theDesc, sizeof(theScale->description) );
		theScale->description[sizeof(theScale->description)-1] = 0;
		strncpy( theScale->unit, theUnit, sizeof(theScale->unit) );
		theScale->unit[sizeof(theScale->unit)-1] = 0;
	}
}

int IM7reader::CreateBuffer(BufferType * myBuffer, int theNX, int theNY, int theNZ, int theNF, int isFloat, int vectorGrid, enum BufferFormat_t imageSubType) {
   if (myBuffer==NULL){
       return 0;
   }
   printf("create attributes: nx %d, ny %d, nz %d, totalLines %d \n", theNX, theNY, theNZ, theNY*theNZ*theNF); 
    
   myBuffer->isFloat = isFloat;
   myBuffer->nx = theNX;
   myBuffer->ny = theNY;
   myBuffer->nz = theNZ;
   myBuffer->nf = theNF;
   myBuffer->totalLines = theNY*theNZ*theNF;
   myBuffer->vectorGrid = vectorGrid;
   myBuffer->image_sub_type = imageSubType;
   
   int size = theNX * myBuffer->totalLines;
   if (isFloat){
      //cudaMallocHost((void**)&(myBuffer->floatArray),sizeof(float)*size);
      myBuffer->floatArray=(float*)malloc(sizeof(float)*size);
   }
   else {
     //myBuffer->wordArray = (Word*)cudaMallocHost(sizeof(Word)*size);
     //printf("I'm a word array\n");
     myBuffer->wordArray=(Word*)malloc(sizeof(Word)*size);
     //cudaMallocHost((void**)&(myBuffer->wordArray),sizeof(Word)*size);
   }
   SetBufferScale( &(myBuffer->scaleX), 1, 0, "", "pixel" );
   SetBufferScale( &(myBuffer->scaleY), 1, 0, "", "pixel" );
   SetBufferScale( &(myBuffer->scaleI), 1, 0, "", "counts" );
   return (myBuffer->floatArray!=NULL || myBuffer->wordArray!=NULL);
}

int IM7reader::ReadIM7 ( const char* theFileName, BufferType* myBuffer, AttributeList** myList )
{
	FILE* theFile = fopen(theFileName, "rb"); // open for binary read
	if (theFile==NULL) {
	   printf("Error Opening File\n");
	   return IMREAD_ERR_FILEOPEN;
	}
	//printf("I'm in IM7\n");
	// Read an image in our own IMX or IMG or VEC or VOL format
	int theNX,theNY,theNZ,theNF;
	// read and store file header contents
	Image_Header_7 header;
	if (!fread ((char*)&header, sizeof(header), 1, theFile))
	{
	    fclose(theFile);
	    perror("Error reading header\n");
	    return IMREAD_ERR_HEADER;
	}
	/*switch (header.version)
	{
		case IMAGE_IMG:
		case IMAGE_IMX:
		case IMAGE_FLOAT:
		case IMAGE_SPARSE_WORD:
		case IMAGE_SPARSE_FLOAT:
        case IMAGE_PACKED_WORD:
		  fclose(theFile);
		  return ReadIMX(theFileName,myBuffer,myList);
	}*/
	if (header.isSparse)
	{
	  fclose(theFile);
	  return IMREAD_ERR_FORMAT;
	}

	theNX = header.sizeX;
	theNY = header.sizeY;
	theNZ = header.sizeZ;
	theNF = header.sizeF;
	if (header.buffer_format > 0)
	{	// vector
		const int compN[] = { 1, 9, 2, 10, 3, 14 };
		theNY *= compN[header.buffer_format];
	}
	CreateBuffer( myBuffer, theNX,theNY,theNZ,theNF, header.buffer_format!=-4/*word*/, header.vector_grid, (enum BufferFormat_t)header.buffer_format );

	ImReadError_t errret = IMREAD_ERR_NO;
	switch (header.pack_type)
	{
		//case IM7_PACKTYPE_IMG:
		//	errret = SCPackUncompressed_Read(theFile,myBuffer);
		//	break;
		case IM7_PACKTYPE_IMX:
			errret = SCPackOldIMX_Read(theFile,myBuffer);
			break;
		//case IM7_PACKTYPE_ZLIB:
		//	errret = SCPackZlib_Read(theFile,myBuffer);
		//	break;
		default:
			errret = IMREAD_ERR_FORMAT;
	}

	/*if (errret==IMREAD_ERR_NO)
	{
		AttributeList* tmpAttrList = NULL;
		AttributeList** useList = (myList!=NULL ? myList : &tmpAttrList);
        ReadImgExtHeader(theFile,useList);
		AttributeList* ptr = *useList;
		while (ptr!=NULL)
		{
			//fprintf(stderr,"%s: %s\n",ptr->name,ptr->value);
			if (strncmp(ptr->name,"_SCALE_",7)==0)
			{
				switch (ptr->name[7])
				{
					case 'X':	Scale_Read( ptr->value, &myBuffer->scaleX );	break;
					case 'Y':	Scale_Read( ptr->value, &myBuffer->scaleY );	break;
					case 'I':	Scale_Read( ptr->value, &myBuffer->scaleI );	break;
				}
			}
			ptr = ptr->next;
		}
    }*/
	fclose(theFile);
	return errret;
}

void IM7reader::Scale_Read( const char*theData, BufferScaleType* theScale )
{
	int pos;
	sscanf(theData,"%f %f%n",&theScale->factor,&theScale->offset,&pos);
	theScale->unit[0] = 0;
	theScale->description[0] = 0;
	if (pos>0)
	{
		while (theData[pos]==' ' || theData[pos]=='\n')
			pos++;
		strncpy( theScale->unit, theData+pos, sizeof(theScale->unit) );
		theScale->unit[sizeof(theScale->unit)-1] = 0;
		pos++;
		while (theData[pos]!=' ' && theData[pos]!='\n' && theData[pos]!='\0')
			pos++;
		while (theData[pos]==' '|| theData[pos]=='\n')
			pos++;
		strncpy( theScale->description, theData+pos, sizeof(theScale->description) );
		theScale->description[sizeof(theScale->description)-1] = 0;
		// cut unit
		pos = 0;
		while (theScale->unit[pos]!=' ' && theScale->unit[pos]!='\n' && theScale->unit[pos]!='\0')
			pos++;
		theScale->unit[pos] = '\0';
	}
}

ImReadError_t IM7reader::SCPackOldIMX_Read( FILE* theFile, BufferType* myBuffer )
{
	enum {BIT8,			// compressed as 8 bit
			BIT4L,		// compressed to 4 bit left nibble
			BIT4R,		// compressed to 4 bit right nibble
			BIT16 }		// n-uncompressed 2-byte pixels
			cmode;		// compression mode
	/*
	compression syntax:
	  BIT8: - At start lastvalue = 0 and cmode = BIT8, i.e. it is assumed
		  that the following bytes are coded with 1 pixel per byte
	  BIT8: - If -128 is encountered, a pixel as a Word (2 bytes) follows.
		  This value is the absolute value of the pixel.
	  BIT8: - If -127 is encountered cmode is set to BIT4, i.e. from now
		  on, each byte contains 2 pixels, each consisting of 4 bits.
		  Thus each new value can take on values between -7 and +7.
		  This is coded as 0x00-0x07 (0-7) and 0x09-0x0F (-7 to -1)
	  BIT8: - If 127 is encountered, then the next byte gives the number
		  of uncompressed 16-bit pixel values, followed by the 2-byte
		  data.
	  BIT4: - Cmode BIT4 is set back to BIT8 if 0x08 is encountered in the
		  right 4 bits.
	*/

	// Yes. Skip the preview image stored after the header
	Word bytecount, count, lastvalue, newvalue = 0;
	signed char *bpageadr, bvalue = 0, newnibble;
	Byte nx, ny, nbytes = 0;
	int bline;
	Word  poge [PAGESIZE];							// input buffer to increase performance
	Word* page = (Word*)&poge;						// by reducing the # of read operations

	if (!fread(&nx,1,1,theFile))  goto errexit;	   // read x-size of preview
	if (!fread(&ny,1,1,theFile))  goto errexit;		// read y-size of preview
	// Preview: The preview includes the upper 8 bit of a restricted number of pixel.
	//		int steps = max( myBuffer->nx / 100 + 1, myBuffer->ny / 100 + 1 );
	//		Byte ny = (myBuffer->ny-1) / steps + 1;
	//		Byte nx = (myBuffer->nx-1) / steps + 1;
	//		Byte *myPreview = malloc(sizeof(Byte)*ny*nx);
	//		for (int y=0; y<ny; y++)
	//			for (int x=0; x<nx; x++)
	//				fread(&myPreview[nx*y+x],1,1,theFile);
	if (fseek( theFile, nx*ny, SEEK_CUR))  goto errexit;	// skip preview

	cmode = BIT8;                               // start with BIT8 mode
	lastvalue = 0;                              // and assuming a previous 0
	if ( (bytecount = ReadNextPage(theFile, page, &bpageadr, &count)) == 0 )  goto errexit;

	for (bline = 0; bline < myBuffer->totalLines; bline++ )
	{
		Word* rowW = &myBuffer->wordArray[myBuffer->nx * bline];
		int i;
		for (i = 0; i < myBuffer->nx; i++ )
		{
			if ( count == bytecount )               // no more bytes left?
				if ( (bytecount = ReadNextPage(theFile, page, &bpageadr, &count)) == 0 )
					goto errexit;	// error 5
			if ( cmode == BIT4L || cmode == BIT8 )  // need new byte
			{
				bvalue = *bpageadr++;                 // new value
				count++;                              // 1 byte processed
			}

			if ( cmode == BIT4R )                   // process right nibble next
			{
				newnibble = bvalue & 0x0F;            // get right nibble
				cmode = BIT4L;                        // next: left nibble
				if ( newnibble == 0x08 )              // change back to BIT8
				{
					cmode = BIT8;
					if ( count == bytecount )           // no more bytes left?
						if ( (bytecount = ReadNextPage(theFile, page, &bpageadr, &count)) == 0 )
							goto errexit;
					bvalue = *bpageadr++;               // new value BIT8-mode
					count++;                            // 1 byte processed
				}
				else
				{
					if ( newnibble & 0x08 ) newnibble |= 0xF0;  // -1 to -7
					newvalue = lastvalue + (long)newnibble;
				}
			}
			else if ( cmode == BIT4L )              // process left nibble next
			{
				newnibble = bvalue >> 4;
				if ( newnibble & 0x08 ) newnibble |= 0xF0;    // -1 to -7
				newvalue = lastvalue + (long)newnibble;       // get left nibble
				cmode = BIT4R;                        // next: right nibble
			}

			if ( cmode == BIT8 )                    // BIT8 compression mode
			{
				switch ( bvalue )
				{
				case -128:                     // exception?
					cmode = BIT16;              // change to BIT16
					nbytes = 1;                 // read only 1 pixel
					break;
				case 127:                      // exception? n-16-bit values; number of bytes follows
					if ( count == bytecount )         // no more bytes left?
						if ( (bytecount = ReadNextPage(theFile, page, &bpageadr, &count)) == 0 )
							goto errexit;
					cmode = BIT16;                    // set cmode correctly
					nbytes = *( (Byte*)bpageadr++ );  // number of bytes
					count++;                          // 1 byte read
					break;
				case -127:                          // change to BIT4 mode
					if ( count == bytecount )         // no more bytes left?
						if ( (bytecount = ReadNextPage(theFile, page, &bpageadr, &count)) == 0 )
							goto errexit;
					bvalue = *bpageadr++;             // get new byte
					count++;
					newnibble = bvalue >> 4;
					if ( newnibble & 0x08 ) newnibble |= 0xF0; // -1 to -7
					newvalue = lastvalue + (long)newnibble;    // get left nibble
					cmode = BIT4R;                    // process right nibble next
					break;
				default:
					newvalue = lastvalue + (long)bvalue;  // get new value, normal mode
				}
			}			// if cmode = BIT8

			if ( cmode == BIT16 )                   // nbytes 16-bit pixels
			{
				if ( count == bytecount )             // no more bytes left?
					if ( (bytecount = ReadNextPage(theFile, page, &bpageadr,	&count)) == 0 )
						goto errexit;		// error 5
				if ( count == bytecount-1 )           // only one byte left?
				{                                     // ---> trouble!
					newvalue = (Word)*(Byte*)bpageadr;  // get low byte first from old page
					if ( (bytecount = ReadNextPage(theFile, page, &bpageadr,	&count)) == 0 )
						goto errexit;		// error 5
					newvalue += ((Word)*(Byte*)bpageadr++) << 8; // get high byte from new page
					count = 1;                          // one byte used already
				}
				else
				{
					newvalue = *( (Word*)bpageadr );    // get new value, Word pointer!
					bpageadr += 2;                      // increment byte pointer
					count += 2;                         // 2 bytes processed
				}
				if ( --nbytes == 0 ) cmode = BIT8;    // decrement counter, change mode
			}

			if ( i < myBuffer->nx )                     // maybe skip right part
				rowW[i] = newvalue;
				//pixbuf.W[bline][i] = newvalue;			// store in pixbuf
			lastvalue = newvalue;                  // update last value

		} 	// for i
	}	// for bline

	fseek( theFile, IMX_pagepos + ((char*)bpageadr - (char*)page), SEEK_SET );
	return IMREAD_ERR_NO;

errexit:
	return IMREAD_ERR_DATA;
}


void IM7reader::CreatePinnedPool() {
    return;
}

int main(void) {
IM7reader image("01.IM7");
return 0;
}

