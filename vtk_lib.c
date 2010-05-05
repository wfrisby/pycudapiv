#include <stdio.h>

struct float2_
{
    float x;
    float y;
};
typedef struct float2_ float2;

void vtk_write_header(FILE *u,char const *title) {
    fprintf(u,"# vtk DataFile Version 2.0\n");
    fprintf(u,"%s\n",title);
    fprintf(u,"ASCII\n");
    fprintf(u,"\n");
}

void vtk_write_grid_2d(FILE *f, int w, int h) {
    int ii,jj;
    fprintf(f,"DATASET STRUCTURED_GRID\n");
    fprintf(f,"DIMENSIONS %d\t%d\t%d\n",w,h,1);
    fprintf(f,"POINTS %d float\n",w*h);
    int step = 1024/w;
    for(ii=h;ii>0;ii--) {
        for(jj=w;jj>0;jj--) {
            fprintf(f,"%f\t%f\t%f\n",(float)jj*step, (float)ii*step, (float)0.0f);
        }
    }
    fprintf(f,"\n");
    fprintf(f,"POINT_DATA \t%d\n", w*h);
}

void vtk_write_vector_2d(FILE *f, char const *name, float2 * vectors, int len) {
    int ii;
    fprintf(f,"VECTORS %s float\n",name);
    for (ii=0;ii<len;vectors++,ii++) {
        fprintf(f,"%f\t%f\t%f\n",vectors->x,vectors->y, (float)0.0f);
    }
}
